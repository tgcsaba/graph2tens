import tensorflow as tf

from tensorflow.keras import layers, callbacks, Model, activations, regularizers

from spektral.layers import GlobalAvgPool, GlobalAttentionPool, GCNConv, GATConv, GraphSageConv,\
                            ChebConv, GINConv

import nn_ops

# ----------------------------------------------------------------------------------------------------

SUPPORTED_GNN_TYPE = [
    'GSAN', 'GSN', 'GCN', 'GAT', 'GraphSage', 'Cheb', 'GIN'
]

NAME_TO_CLASS_SPEKTRAL = {
    'GCN' : GCNConv,
    'GAT' : GATConv,
    'GraphSage' : GraphSageConv,
    'Cheb' : ChebConv,
    'GIN' : GINConv
}

# ----------------------------------------------------------------------------------------------------

def _norm_layer(norm_type, graph_mode=True, single_mode=False):
    if norm_type == 'layer' or not graph_mode:
        norm_layer = layers.LayerNormalization(center=False, scale=False)
    elif norm_type == 'graph':
        if single_mode:
            norm_layer = layers.Lambda(lambda x: (x - tf.reduce_mean(x, axis=(0, -1), keepdims=True)) \
                / tf.maximum(tf.math.reduce_std(x, axis=(0, -1), keepdims=True), 1e-6))
        else:
            norm_layer = nn_ops.layers.GraphNorm()
    return norm_layer

# ----------------------------------------------------------------------------------------------------

class GNNModel(Model):
    def __init__(
        self, n_labels, gnn_type='GSAN', n_features=32, n_depth=2, n_levels=2, n_steps=2, n_heads=1,
        K=1, epsilon=0., diff=True, basepoint=True, norm_type='', jk_type='jk_max', dropout=None,
        l2_reg=None, attn_dropout=None, learn_embedding=True, use_proj=False, n_pre=0, n_post=0,
        n_pre_dim=None, n_post_dim=None, pre_dropout=None, post_dropout=None, pre_skip=False,
        post_skip=False, pool='attn', node_level=False, single_mode=False, pred='multi'):
        
        # Validate some categorical parameters
        gnn_type = gnn_type.lower()
        gnn_type = gnn_type.split(',')
        if len(gnn_type) > 2:
            raise ValueError('Only up to 2 GNN stacks can be defined.')
        for gtype in gnn_type:
            if gtype not in [x.lower() for x in SUPPORTED_GNN_TYPE]:
                raise ValueError(
                    f'Unsupported base GNN \'{gtype}\'. \
                        Only support {SUPPORTED_GNN_TYPE}')
        
        jk_type = jk_type.lower() if jk_type is not None else None
        if jk_type is not None and jk_type not in ['jk_cat', 'jk_max']:
            raise ValueError(
                f'Unsupported jk-type \'{jk_type}\'. \
                    Only support \'jk_cat\, \'jk_max\' or None')
            
        pool = pool.lower()
        if pool not in ['attn', 'avg']:
            raise ValueError(
                f'Unsupported pool-type \'{pool}\'. \
                    Only support \'attn\' or \'avg\'')
        
        pred = pred.lower()
        if pred not in ['multi', 'binary', 'linear']:
            raise ValueError(
                f'Unsupported pred-type \'{pred}\'. \
                    Only support \'multi\', \'binary\' or \'linear\'')
            
        
            
        super().__init__()
        
        self.gnn_type = gnn_type
        self.n_stacks = len(self.gnn_type)
        
        self.n_features = n_features
        self.n_depth =  n_depth
        self.n_labels = n_labels
        self.n_levels = n_levels
        self.n_steps = n_steps
        self.n_heads = n_heads
        self.K = K
        self.epsilon = epsilon
        self.diff = diff
        self.basepoint = basepoint
        self.learn_embedding = learn_embedding
        self.use_proj = use_proj
        
        self.norm_type = norm_type
        self.jk_type = jk_type
        
        self.attn_dropout = attn_dropout
        self.dropout = dropout
        self.l2_reg = l2_reg
        
        self.n_pre = n_pre
        self.n_post = n_post
        self.n_pre_dim = n_pre_dim or n_features
        self.n_post_dim = n_post_dim or n_features
        self.pre_dropout = pre_dropout or dropout
        self.post_dropout = post_dropout or dropout
        self.pre_skip = pre_skip
        self.post_skip = post_skip
        
        self.pool = pool
        self.single_mode = single_mode
        self.node_level = node_level
        self.pred = pred
        
        self.use_pre = n_pre is not None and n_pre > 0
        self.use_post = n_post is not None and n_post > 0
        self.use_dp = dropout is not None and dropout > 0.
        self.use_pre_dp = self.pre_dropout is not None and self.pre_dropout > 0.
        self.use_post_dp = self.post_dropout is not None and self.post_dropout > 0.
        
        self.use_norm = norm_type == 'layer' or norm_type == 'graph'
        self.use_l2_reg = l2_reg is not None and l2_reg > 0.
        self.use_jk = self.jk_type is not None
        self.use_tens = [
            self.gnn_type[i] == 'gsn' or self.gnn_type[i] == 'gsan'
            for i in range(len(self.gnn_type))]
        
        # set up l2 regularizers
        regularizer = regularizers.l2(l2_reg) if self.use_l2_reg else None
        if any(self.use_tens):
            tens_regularizer = nn_ops.regularizers.Rank1TensorL2(l2_reg) if self.use_l2_reg else None
# 
        # preprocessing layers
        if self.use_pre:
            self.premlp = []
            if self.use_norm:
                self.prenorm = []
            if self.use_pre_dp:
                self.pre_dp_layer = []
            if self.pre_skip:
                self.preproj = layers.Dense(self.n_pre_dim)
                self.preskip = []
            for i in range(self.n_pre):
                self.premlp.append(
                    layers.Dense(self.n_pre_dim, 'relu', kernel_regularizer=regularizer))
                if self.use_norm:
                    self.prenorm.append(_norm_layer(self.norm_type, single_mode=single_mode))
                if self.use_pre_dp:
                    self.pre_dp_layer.append(layers.Dropout(self.pre_dropout))
                if self.pre_skip:
                    self.preskip.append(layers.Add())
        
        # postprocessing layers
        if self.use_post:
            if self.use_post_dp:
                self.post_dp_layer = []
            self.postmlp = []
            if self.use_norm:
                self.postnorm = []
            if self.post_skip:
                self.postproj = layers.Dense(self.n_post_dim)
                self.postskip = []
            for i in range(self.n_post):
                self.postmlp.append(
                    layers.Dense(self.n_post_dim, 'relu', kernel_regularizer=regularizer))
                if self.use_norm:
                    self.postnorm.append(_norm_layer(
                        self.norm_type, single_mode=single_mode, graph_mode=self.node_level))
                if self.use_post_dp:
                    self.post_dp_layer.append(layers.Dropout(self.post_dropout))
                if self.post_skip:
                    self.postskip.append(layers.Add())
    
        if self.use_dp:
            self.graph_dp = [[] for _ in range(self.n_stacks)]
            
        # GNN layers
        self.graph_layer = [[] for _ in range(self.n_stacks)]
        
        if self.use_norm:
            self.norm = [[] for _ in range(self.n_stacks)]
        
        if any(self.use_tens):
            self.flat = [[] for _ in range(self.n_stacks)]
            if self.use_proj:
                if self.use_dp:
                    self.proj_dp = [[] for _ in range(self.n_stacks)]
                self.proj = [[] for _ in range(self.n_stacks)]
            if self.use_norm:
                if self.norm_type == 'layer':
                    self.rescale = [[] for _ in range(self.n_stacks)]
                if self.use_proj:
                    self.proj_norm = [[] for _ in range(self.n_stacks)]
                    
        for j in range(self.n_stacks):
            for i in range(self.n_depth):
                if self.gnn_type[j] == 'gsan':
                    g = nn_ops.layers.GSAN(
                        self.n_features//self.n_heads, self.n_levels, self.n_steps, diff=self.diff,
                        n_heads=self.n_heads, dropout=self.attn_dropout, basepoint=self.basepoint,
                        learn_embedding=self.learn_embedding, kernel_regularizer=tens_regularizer)
                elif self.gnn_type[j] == 'gsn':
                    g = nn_ops.layers.GSN(
                        self.n_features, self.n_levels, self.n_steps, basepoint=self.basepoint,
                        diff=self.diff, kernel_regularizer=tens_regularizer,
                        learn_embedding=self.learn_embedding)
                elif self.gnn_type[j] == 'gcn':
                    g = GCNConv(self.n_features, 'relu', kernel_regularizer=regularizer)
                elif self.gnn_type[j] == 'gat':
                    g = GATConv(
                        self.n_features//self.n_heads, self.n_heads, activation='elu',
                        dropout_rate=self.attn_dropout, kernel_regularizer=regularizer)
                    # use ELU activation as suggested in https://arxiv.org/pdf/1710.10903.pdf
                elif self.gnn_type[j] == 'graphsage':
                    g = GraphSageConv(
                        self.n_features, 'max', activation='relu', kernel_regularizer=regularizer)
                    # use max-aggregator and ReLU activation
                    # as suggested in https://arxiv.org/pdf/1706.02216.pdf
                elif self.gnn_type[j] == 'cheb':
                    g = ChebConv(
                        self.n_features, K=self.K, activation='relu',
                        kernel_regularizer=regularizer)
                elif self.gnn_type[j] == 'gin':
                    g = GINConv(
                        self.n_features, epsilon=self.epsilon, activation='relu',
                        kernel_regularizer=regularizer)
                self.graph_layer[j].append(g)
                if self.use_norm:
                    self.norm[j].append(_norm_layer(self.norm_type, single_mode=single_mode))
                if self.use_dp:
                    self.graph_dp[j].append(layers.Dropout(self.dropout))
                if self.use_tens[j]:
                    n_features_ = (self.n_features // self.n_heads) * self.n_heads \
                                        if self.gnn_type[j] == 'gsan' else self.n_features
                    self.flat[j].append(layers.Reshape((n_features_ * self.n_levels,)))
                    if self.use_proj:
                        if self.use_dp:
                            self.proj_dp[j].append(layers.Dropout(self.dropout))
                        self.proj[j].append(layers.Dense(self.n_features))
                    if self.use_norm:
                        if self.norm_type == 'layer':
                            self.rescale[j].append(layers.Rescaling(1./self.n_levels))
                        if self.use_proj:
                            self.proj_norm[j].append(_norm_layer(self.norm_type, single_mode=single_mode))
        
        if self.jk_type == 'jk_cat':
            self.jk_pool = [layers.Concatenate() for _ in range(self.n_stacks)]
        elif self.jk_type == 'jk_max':
            self.jk_pool = [layers.Maximum() for _ in range(self.n_stacks)]
            
        if self.n_stacks > 1:
            self._n_features = [self.n_features if self.use_proj else
                                (self.n_features // self.n_heads) * self.n_heads * self.n_levels
                                if self.gnn_type[j] == 'gsan' else self.n_features * self.n_levels
                                if self.gnn_type[j] == 'gsn' else self.n_features
                                for j in range(self.n_stacks)]
            self.stack_proj = [layers.Dense(self._n_features[j]) for j in range(self.n_stacks)]
            self.stack_add = [layers.Add() for _ in range(self.n_stacks)]
        
        if not self.node_level:
            if self.pool == 'attn':
                self.pool = GlobalAttentionPool(self.n_post_dim)
            elif self.pool == 'avg':
                self.pool = GlobalAvgPool()
            if self.use_norm:
                self.poolnorm = _norm_layer(
                    self.norm_type, single_mode=single_mode, graph_mode=False)
                
        self.pred = layers.Dense(self.n_labels, 'softmax')

    def call(self, inputs):
        
        x, a = inputs[0], inputs[1]
        if not self.single_mode:
            i = inputs[-1]
        
        out = x
        
        if self.use_pre:
            for j in range(self.n_pre):
                out_in = out
                if self.pre_skip and j==0:
                    out_in = self.preproj(out_in)
                out = self.premlp[j](out)
                if self.use_norm:
                    out = self.prenorm[j]([out, i]) \
                        if self.norm_type == 'graph' and not self.single_mode else self.prenorm[j](out)
                if self.use_pre_dp:
                    out = self.pre_dp_layer[j](out)
                if self.pre_skip:
                    out = self.preskip[j]([out_in, out])
                    
        if self.n_stacks > 1:
            stack1_skip = self.stack_proj[0](out)
                
        for k in range(self.n_stacks):
            if k == 1:
                out = self.stack_add[0]([stack1_skip, out])
            if self.use_jk:
                jk_skip = []
            for j in range(self.n_depth):
                out = self.graph_layer[k][j]([out, a])
                if self.use_norm:
                    out = self.norm[k][j]([out, i]) \
                        if self.norm_type == 'graph' and not self.single_mode else self.norm[k][j](out)
                    if self.use_tens[k] and self.norm_type == 'layer':
                        out = self.rescale[k][j](out)
                if self.use_tens[k]:
                    out = self.flat[k][j](out)
                    if self.use_proj:
                        if self.use_dp:
                            out = self.proj_dp[k][j](out)
                        out = self.proj[k][j](out)
                        if self.use_norm:
                            out = self.proj_norm[k][j]([out, i]) \
                                if self.norm_type == 'graph' and not self.single_mode else self.proj_norm[k][j](out)
                if self.use_jk:
                    jk_skip.append(out)
                if self.use_dp:
                    out = self.graph_dp[k][j](out)    
            if self.use_jk:
                out = self.jk_pool[k](jk_skip)
            if self.n_stacks > 1 and k == 0:
                stack2_skip = self.stack_proj[1](out)
        
        if self.n_stacks > 1:
            out = self.stack_add[1]([stack2_skip, out])
        
        if not self.node_level:
            out = self.pool([out, i])
            if self.use_norm:
                out = self.poolnorm(out)
                
        if self.use_post:
            for j in range(self.n_post):
                out_in = out
                if self.post_skip and j==0:
                    out_in = self.postproj(out_in)
                out = self.postmlp[j](out)
                if self.use_norm:
                    out = self.postnorm[j]([out, i]) \
                        if self.norm_type == 'graph' and not self.single_mode and self.node_level \
                        else self.postnorm[j](out)
                if self.use_post_dp:
                    out = self.post_dp_layer[j](out)
                if self.post_skip:
                    out = self.postskip[j]([out_in, out])
        out = self.pred(out)  
        return out
    
# ----------------------------------------------------------------------------------------------------