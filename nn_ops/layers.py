import numpy as np
import tensorflow as tf

from .initializers import FactorialInit, GSNUniformInit, GSNNormalInit
from .regularizers import Rank1TensorL1, Rank1TensorL2
from .algorithms import graph_attn_prob, graph_sig_network, graph_sig_attn_network

from tensorflow.keras import initializers, regularizers, constraints, layers

from spektral.layers.ops import add_self_loops_indices

# ----------------------------------------------------------------------------------------------------

class GraphNorm(layers.Layer):
    """GraphNorm for disjoint mode.
    In single mode can just use LayerNorm with correctly specified axes.
    """
    def __init__(self, graph_axis=0, axis=-1, epsilon=1e-3, **kwargs):
        super().__init__(**kwargs)
    
        self.graph_axis = graph_axis
        self.axis = axis
        self.epsilon = epsilon
        
    def call(self, inputs):
        assert len(inputs) == 2
        x, i = inputs
        
        n_dim = tf.shape(x)[self.axis]
        
        # center x
        x -= tf.gather(tf.math.segment_mean(x, i), i, axis=0)
        
        # compute sum of squared entries
        ssq = tf.reduce_sum(
            tf.math.segment_sum(tf.math.square(x), i),
            axis=self.axis, keepdims=True)
        
        # compute number of samples per statistic
        n = tf.math.segment_sum(tf.ones_like(i, dtype=x.dtype), i) * tf.cast(n_dim, x.dtype)
        # compute var
        var = ssq / (n[[...] + [None] * (ssq.shape.ndims-1)] - 1.)
        
        # normalize x
        x /= tf.gather(tf.math.sqrt(var + self.epsilon), i, axis=0)
        return x
        
    def compute_output_shape(self, input_shape):
        return input_shape

# ----------------------------------------------------------------------------------------------------

class GSAN(layers.Layer):
    def __init__(
        self, n_features, n_levels, n_steps, n_heads=1, diff=True, use_bias=True, basepoint=True,
        dropout=None, learn_embedding=True, add_self_loops=False, kernel_initializer='gsn_uniform',
        kernel_regularizer=None, kernel_constraint=None, bias_initializer='zeros',
        bias_regularizer=None, bias_constraint=None, attn_kernel_initializer='glorot_uniform',
        attn_kernel_regularizer=None, attn_kernel_constraint=None, activity_regularizer=None,
        **kwargs):
        """
        Graph Signature Attention Network layer.
        """
    
        super().__init__(activity_regularizer=activity_regularizer, **kwargs)
        
        self.n_features = n_features
        self.n_levels = n_levels
        self.n_steps = n_steps
        self.n_heads = n_heads
        self.dropout = dropout
        self.diff = diff
        self.use_bias = use_bias
        self.basepoint = basepoint
        self.learn_embedding = learn_embedding
        self.add_self_loops = add_self_loops
        
        if kernel_initializer.lower().replace('_', '') == 'gsnuniform':
            self.kernel_initializer = GSNUniformInit(n_levels)
        elif kernel_initializer.lower().replace('_', '') == 'gsnnormal':
            self.kernel_initializer = GSNNormalInit(n_levels)
        else:
            self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        
        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)
            
        if use_bias:
            self.bias_initializer = initializers.get(bias_initializer)
            self.bias_regularizer = regularizers.get(bias_regularizer)
            self.bias_constraint = constraints.get(bias_constraint)
        
        if learn_embedding:
            self.embed_init = FactorialInit(n_levels)

    def build(self, input_shape):
        
        if len(input_shape) == 3:        
            self.has_edge_feat = True
            self.n_edge_dim = input_shape[2][-1]
        elif len(input_shape) == 2:
            self.has_edge_feat = False
        
        self.n_dim = input_shape[0][-1]
        
        
        self.kernel = self.add_weight(
            shape=(self.n_levels, self.n_dim, self.n_heads * self.n_features,), name='kernel',
            initializer=self.kernel_initializer, regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint, dtype=self.dtype, trainable=True)
        
        self.src_kernel = self.add_weight(
            shape=(self.n_dim, self.n_heads,), name='src_kernel',
            initializer=self.attn_kernel_initializer,
            regularizer=self.attn_kernel_regularizer, constraint=self.attn_kernel_constraint,
            dtype=self.dtype, trainable=True)

        self.trg_kernel = self.add_weight(
            shape=(self.n_dim, self.n_heads,), name='trg_kernel',
            initializer=self.attn_kernel_initializer, regularizer=self.attn_kernel_regularizer,
            constraint=self.attn_kernel_constraint, dtype=self.dtype, trainable=True)
        
        if self.has_edge_feat:
            self.edge_kernel = self.add_weight(
                shape=(self.n_edge_dim, self.n_heads,), name='edge_kernel',
                initializer=self.attn_kernel_initializer, regularizer=self.attn_kernel_regularizer, 
                constraint=self.attn_kernel_constraint, dtype=self.dtype, trainable=True)
        
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.n_levels, self.n_heads, self.n_features,), name='bias',
                initializer=self.bias_initializer, regularizer=self.bias_regularizer,
                constraint=self.bias_constraint, dtype=self.dtype, trainable=True)
        
        if self.learn_embedding:
            self.embed_coeffs = self.add_weight(
                shape=(self.n_levels, self.n_heads, self.n_features,), name='embed_coeffs', 
                initializer=self.embed_init, dtype=self.dtype, trainable=True)
        
        super().build(input_shape)
        
    def call(self, inputs, mask=None):
        
        if self.has_edge_feat:
            assert len(inputs) == 3        
            x, a, e = inputs
        else:
            assert len(inputs) == 2        
            x, a = inputs
            
        if x.shape.ndims != 2 and x.shape.ndims != 3:
            raise ValueError('Only single/disjoint/batch mode are supported.')
        
        # get list of edge indices
        if isinstance(a, tf.SparseTensor):
            # disjoint/single mode
            edge_idx = a.indices
            if self.add_self_loops:
                n_total_nodes = tf.shape(x, out_type=edge_idx.dtype)[0]
                edge_idx = add_self_loops_indices(edge_idx, n_total_nodes)
        else:
            # batch mode
            if self.add_self_loops:
                shape = tf.shape(a)[:-1]
                a = tf.linalg.set_diag(a, tf.ones(shape, a.dtype))
            edge_idx = tf.where(a != 0)
            
        kernel_ = tf.reshape(
            self.kernel, (self.n_levels, self.n_dim, self.n_heads, self.n_features))
        
        attn_prob = graph_attn_prob(
            x, edge_idx, self.src_kernel, self.trg_kernel, dropout=self.dropout,
            edge_feat=e if self.has_edge_feat else None,
            edge_kernel=self.edge_kernel if self.has_edge_feat else None)
            
        feat = graph_sig_attn_network(
            x, edge_idx, attn_prob, self.n_levels, self.n_steps, kernel_, diff=self.diff,
            basepoint=self.basepoint, bias=self.bias if self.use_bias else None,
            embed_coeffs=self.embed_coeffs if self.learn_embedding else None, dtype=self.dtype)
        
        return feat
            
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_levels, self.n_heads * self.n_features)

# ----------------------------------------------------------------------------------------------------

class GSN(layers.Layer):
    def __init__(
        self, n_features, n_levels, n_steps, diff=True, use_bias=True, basepoint=True,
        learn_embedding=True, add_self_loops=True, kernel_initializer='gsn_uniform',
        kernel_regularizer=None, kernel_constraint=None, bias_initializer='zeros',
        bias_regularizer=None, bias_constraint=None, activity_regularizer=None, **kwargs):
        """
        Graph Signature Network layer.
        """
    
        super().__init__(activity_regularizer=activity_regularizer, **kwargs)
        
        self.n_features = n_features
        self.n_levels = n_levels
        self.n_steps = n_steps
        self.diff = diff
        self.use_bias = use_bias
        self.basepoint = basepoint
        self.learn_embedding = learn_embedding
        self.add_self_loops = add_self_loops
        
        if kernel_initializer.lower().replace('_', '') == 'gsnuniform':
            self.kernel_initializer = GSNUniformInit(n_levels)
        elif kernel_initializer.lower().replace('_', '') == 'gsnnormal':
            self.kernel_initializer = GSNNormalInit(n_levels)
        else:
            self.kernel_initializer = initializers.get(kernel_initializer)
            
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
            
        if use_bias:
            self.bias_initializer = initializers.get(bias_initializer)
            self.bias_regularizer = regularizers.get(bias_regularizer)
            self.bias_constraint = constraints.get(bias_constraint)
            
        if learn_embedding:
            self.embed_init = FactorialInit(n_levels)       

    def build(self, input_shape):
        self.n_dim = input_shape[0][-1]
        
        self.n_components = self.n_levels
        
        self.kernel = self.add_weight(
            shape=(self.n_components, self.n_dim, self.n_features,), name='kernel',
            initializer=self.kernel_initializer, regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint, dtype=self.dtype, trainable=True)
        
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.n_components, self.n_features,), name='bias',
                initializer=self.bias_initializer, regularizer=self.bias_regularizer,
                constraint=self.bias_constraint, dtype=self.dtype, trainable=True)
            
        if self.learn_embedding:
            self.embed_coeffs = self.add_weight(
                shape=(self.n_levels, self.n_features,), name='embed_coeffs',
                initializer=self.embed_init, dtype=self.dtype, trainable=True)
        
        super().build(input_shape)
        
    def call(self, inputs, mask=None):
        x, a = inputs
        if x.shape.ndims != 2 and x.shape.ndims != 3:
            raise ValueError('Only single/disjoint/batch mode are supported.')
        
        # get list of edge indices
        if isinstance(a, tf.SparseTensor):
            # disjoint/single mode
            edge_idx = a.indices
            if self.add_self_loops:
                n_total_nodes = tf.shape(x, out_type=edge_idx.dtype)[0]
                edge_idx = add_self_loops_indices(edge_idx, n_total_nodes)
        else:
            # batch mode
            if self.add_self_loops:
                shape = tf.shape(a)[:-1]
                a = tf.linalg.set_diag(a, tf.ones(shape, a.dtype))
            edge_idx = tf.where(a != 0)
            
        outputs = graph_sig_network(
            x, edge_idx, self.n_levels, self.n_steps, self.kernel, diff=self.diff,
            basepoint=self.basepoint, bias=self.bias if self.use_bias else None,
            embed_coeffs=self.embed_coeffs if self.learn_embedding else None, dtype=self.dtype)
        
        return outputs
            
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_levels, self.n_features)
    
# ----------------------------------------------------------------------------------------------------