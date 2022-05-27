import numpy as np
import tensorflow as tf

TF_INT_TYPE = tf.int64
TF_FLOAT_TYPE = tf.float64
JITTER_LEVEL = 1e-6

# ----------------------------------------------------------------------------------------------------

@tf.function
def graph_attn_prob(
    inp, edge_idx, src_kernel, trg_kernel, edge_feat=None, edge_kernel=None, dropout=None, dtype=None):
    """
    Implements additive (Bahdanau) neighbourhood graph attention. 
    """
    
    batch_mode = inp.shape.ndims == 3
    
    n_dim = tf.shape(inp)[-1]
    
    inp_ = tf.reshape(inp, (-1, n_dim))
    n_total_nodes = tf.shape(inp_)[0]
    
    dtype_ = dtype or TF_FLOAT_TYPE
    if dtype is not None:
        assert inp.dtype == dtype
    dtype_ = inp.dtype
    
    if batch_mode:
        src_idx = tf.cast(n_nodes, TF_INT_TYPE) * edge_idx[:, 0] +  edge_idx[:, 1]
        trg_idx = tf.cast(n_nodes, TF_INT_TYPE) * edge_idx[:, 0] +  edge_idx[:, 2]
    else:
        src_idx = edge_idx[:, 0]
        trg_idx = edge_idx[:, 1]
    
    # compute attention source and target
    src = tf.matmul(inp_, src_kernel)
    trg = tf.matmul(inp_, trg_kernel)

    # calculate transition probabilities using robust softmax attention
    attn_weights = tf.gather(src, src_idx, axis=0) + tf.gather(trg, trg_idx, axis=0)
    if edge_feat is not None and edge_kernel is not None:
        attn_weights += tf.matmul(edge_feat, edge_kernel)
    attn_weights = tf.nn.leaky_relu(attn_weights, alpha=0.2)
        
    # subtract max from each, does not change the softmax but avoids overflow errors
    attn_weights -= tf.gather(
        tf.math.unsorted_segment_max(attn_weights, src_idx, n_total_nodes), 
        src_idx, axis=0)
    attn_prob = tf.math.exp(attn_weights)
    
    # randomly drop connections
    if dropout is not None and dropout > 0.:
        dropout_mask = tf.nn.dropout(tf.ones_like(attn_prob), dropout)
    
    attn_prob /= tf.maximum(tf.gather(
        tf.math.unsorted_segment_sum(attn_prob, src_idx, n_total_nodes), 
        src_idx, axis=0), JITTER_LEVEL)
    return attn_prob

# ----------------------------------------------------------------------------------------------------

@tf.function
def graph_sig_attn_network(inp, edge_idx, attn_prob, n_levels, n_steps, kernel, diff=True,
                           bias=None, embed_coeffs=None, basepoint=True, dtype=None):
    
    assert inp.shape.ndims == 2 or inp.shape.ndims == 3
    assert inp.shape.ndims == edge_idx.shape[1]
    
    batch_mode = inp.shape.ndims == 3    
    
    dtype_ = dtype or TF_FLOAT_TYPE
    if dtype is not None:
        assert inp.dtype == dtype
    dtype_ = inp.dtype
    
    n_edges = tf.shape(edge_idx)[0]
    n_heads, n_features = tf.unstack(tf.shape(kernel)[-2:])
    
    n_nodes, n_dim = tf.unstack(tf.shape(inp)[-2:])
    
    inp_ = tf.reshape(inp, (-1, n_dim))
    n_total_nodes = tf.shape(inp_)[0]
    
    # get for each edge the start and end indices
    if batch_mode:
        src_idx = tf.cast(n_nodes, TF_INT_TYPE) * edge_idx[:, 0] +  edge_idx[:, 1]
        trg_idx = tf.cast(n_nodes, TF_INT_TYPE) * edge_idx[:, 0] +  edge_idx[:, 2]
    else:
        src_idx = edge_idx[:, 0]
        trg_idx = edge_idx[:, 1]
    
    # kernel is (n_levels, n_dim, n_heads, n_features)
    # inp is (n_total_nodes, n_dim)
    inp_proj = tf.einsum('id,mdhp->mihp', inp_, kernel)
    # (n_levels, n_total_nodes, n_heads, n_features)
    # put levels axis first, as tf.map_fn stacks along the first axis
    if bias is not None:
        # bias is (n_levels, n_heads, n_features)
        inp_proj += bias[:, None, :, :]
    if diff:
        update = tf.gather(inp_proj, trg_idx, axis=1) - tf.gather(inp_proj, src_idx, axis=1)
    else:
        update = tf.gather(inp_proj, src_idx, axis=1)
    
    if embed_coeffs is None:
        fact = tf.math.cumprod(tf.range(1, n_levels+1, dtype=dtype_))
        embed_coeffs = 1. / fact[:, None, None]
    
    if diff:
        _update = tf.math.cumprod(update[::-1], axis=0) * attn_prob[None, :, :, None]
        feat = tf.map_fn(
            lambda m: tf.math.unsorted_segment_sum(_update[m], src_idx, n_total_nodes),
            tf.range(n_levels), parallel_iterations=n_levels,
            fn_output_signature=dtype_) * embed_coeffs[:, None] 
    else:
        feat = tf.math.cumprod(inp_proj[::-1], axis=0) * embed_coeffs[:, None]
    
    def _compute_level(m, feat_prev):
        update_cprod = tf.math.cumprod(update[n_levels-1-m:], axis=0) \
                       * embed_coeffs[:m+1, None]

        feat_m = tf.math.unsorted_segment_sum(
            tf.gather(feat_prev[m], trg_idx, axis=0) * attn_prob[:, :, None],
            src_idx, n_total_nodes)
        
        feat_m += tf.math.unsorted_segment_sum(
            update_cprod[-1] * attn_prob[:, :, None],
            src_idx, n_total_nodes)

        def _true_fn():
            feat_prev_rev = feat_prev[m-1::-1]
            _update = tf.math.unsorted_segment_sum(
                tf.reduce_sum(update_cprod[:-1]
                              * tf.gather(feat_prev_rev, trg_idx, axis=1), axis=0)
                * attn_prob[:, :, None],
                src_idx, n_total_nodes)
            return _update
        _false_fn = lambda: tf.zeros(tuple(), dtype=dtype_)
        feat_m += tf.cond(tf.greater(m, 0), _true_fn, _false_fn)
        return feat_m
        
    def _compute_all_levels(feat_prev):
        return tf.map_fn(lambda m:_compute_level(m, feat_prev),
                         tf.range(n_levels), parallel_iterations=n_levels,
                         fn_output_signature=dtype_)
    
    for l in range(int(diff), n_steps):
        feat = _compute_all_levels(feat)
        
    def _add_basepoint_m(m, feat_prev):
        inp_proj_cprod = tf.math.cumprod(inp_proj[n_levels-1-m:], axis=0) \
                         * embed_coeffs[:m+1, None]
        feat_m = feat_prev[m] + inp_proj_cprod[-1]
        def _true_fn():
            feat_prev_rev = feat_prev[m-1::-1]
            _update = tf.reduce_sum(inp_proj_cprod[:-1] * feat_prev_rev, axis=0)
            return _update
        _false_fn = lambda: tf.zeros(tuple(), dtype=dtype_)
        feat_m += tf.cond(tf.greater(m, 0), _true_fn, _false_fn)
        return feat_m
    
    def _add_basepoint(feat_prev):
        return tf.map_fn(lambda m: _add_basepoint_m(m, feat_prev),
                         tf.range(n_levels), parallel_iterations=n_levels,
                         fn_output_signature=dtype_)
        
    # add base-point augmentation
    if diff and basepoint:
        feat = _add_basepoint(feat)
        
    feat = tf.transpose(feat, (1, 0, 2, 3))
    feat = tf.reshape(feat, (-1,) * int(batch_mode) + (n_nodes, n_levels, n_heads * n_features))
    return feat

# ----------------------------------------------------------------------------------------------------

@tf.function
def graph_sig_network(inp, edge_idx, n_levels, n_steps, kernel, diff=True, bias=None, 
                      transition_weights=None, embed_coeffs=None, basepoint=True, dtype=None):
    
    assert inp.shape.ndims == 2 or inp.shape.ndims == 3
    assert inp.shape.ndims == edge_idx.shape[1]
    
    batch_mode = inp.shape.ndims == 3    
    
    dtype_ = dtype or TF_FLOAT_TYPE
    if dtype is not None:
        assert inp.dtype == dtype
        dtype_ = inp.dtype
    else:
        dtype_ = inp.dtype
    
    n_edges = tf.shape(edge_idx)[0]
    n_features = tf.shape(kernel)[-1]
    
    n_nodes, n_dim = tf.unstack(tf.shape(inp)[-2:])
    
    inp_ = tf.reshape(inp, (-1, n_dim))
    n_total_nodes = tf.shape(inp_)[0]
    
    # get for each edge the index of outgoing node and incoming node
    if batch_mode:
        src_idx = tf.cast(n_nodes, TF_INT_TYPE) * edge_idx[:, 0] +  edge_idx[:, 1]
        trg_idx = tf.cast(n_nodes, TF_INT_TYPE) * edge_idx[:, 0] +  edge_idx[:, 2]
    else:
        src_idx = edge_idx[:, 0]
        trg_idx = edge_idx[:, 1]
    
    # kernel is (n_levels, n_dim, n_features)
    # inp is (n_total_nodes, n_dim)
    inp_proj = tf.einsum('id,mdp->mip', inp_, kernel)
    # (n_levels, n_total_nodes, n_features)
    if bias is not None:
        # bias is (n_levels, n_features)
        inp_proj += bias[:, None]
    if diff:
        update = tf.gather(inp_proj, trg_idx, axis=1) - tf.gather(inp_proj, src_idx, axis=1)
    else:
        update = tf.gather(inp_proj, src_idx, axis=1)
        
    degrees = tf.maximum(
        tf.math.unsorted_segment_sum(tf.ones((n_edges,), dtype=dtype_), src_idx, n_total_nodes),
        JITTER_LEVEL)
    
    if embed_coeffs is None:
        fact = tf.math.cumprod(tf.range(1, n_levels+1, dtype=dtype_))
        embed_coeffs = 1. / fact[:, None]
        
    n_steps_ = n_steps + int(not diff)
        
    if diff:
        _update = tf.math.cumprod(update[::-1], axis=0)
        feat = tf.map_fn(
            lambda m: tf.math.unsorted_segment_sum(_update[m], src_idx, n_total_nodes),
            tf.range(n_levels), parallel_iterations=n_levels,
            fn_output_signature=dtype_) * embed_coeffs[:, None]
        feat /= degrees[None, :, None]
    else:
        feat = tf.math.cumprod(inp_proj[::-1], axis=0) * embed_coeffs[:, None]
    
    def _compute_level(m, feat_prev):
        update_cprod = tf.math.cumprod(update[n_levels-1-m:], axis=0) \
                       * embed_coeffs[:m+1, None]

        feat_m = tf.math.unsorted_segment_sum(
            tf.gather(feat_prev[m], trg_idx, axis=0),
            src_idx, n_total_nodes)
        
        feat_m += tf.math.unsorted_segment_sum(
            update_cprod[-1],
            src_idx, n_total_nodes)

        def _true_fn():
            feat_prev_rev = feat_prev[m-1::-1]
            _update = tf.math.unsorted_segment_sum(
                tf.reduce_sum(update_cprod[:-1] * tf.gather(feat_prev_rev, trg_idx, axis=1), axis=0),
                src_idx, n_total_nodes)
            return _update
        _false_fn = lambda: tf.zeros(tuple(), dtype=dtype_)
        feat_m += tf.cond(tf.greater(m, 0), _true_fn, _false_fn)
        return feat_m
        
    def _compute_all_levels(feat_prev):
        return tf.map_fn(lambda m:_compute_level(m, feat_prev),
                         tf.range(n_levels), parallel_iterations=n_levels,
                         fn_output_signature=dtype_)
    
    for l in range(1, n_steps_):
        feat = _compute_all_levels(feat) / degrees[None, :, None]
        
    def _add_basepoint_m(m, feat_prev):
        update_cprod = tf.math.cumprod(inp_proj[n_levels-1-m:], axis=0) \
                       * embed_coeffs[:m+1, None]
        feat_m = feat_prev[m] + update_cprod[-1]
        def _true_fn():
            feat_prev_rev = feat_prev[m-1::-1]
            _update = tf.reduce_sum(update_cprod[:-1] * feat_prev_rev, axis=0)
            return _update
        _false_fn = lambda: tf.zeros(tuple(), dtype=dtype_)
        feat_m += tf.cond(tf.greater(m, 0), _true_fn, _false_fn)
        return feat_m
    
    def _add_basepoint(feat_prev):
        return tf.map_fn(lambda m: _add_basepoint_m(m, feat_prev),
                         tf.range(n_levels), parallel_iterations=n_levels,
                         fn_output_signature=dtype_)
        
    # add base-point augmentation
    if diff and basepoint:
        feat = _add_basepoint(feat)
            
    feat = tf.transpose(feat, (1, 0, 2))
    feat = tf.reshape(feat, (-1,) * int(batch_mode) + (n_nodes,) + (n_levels, n_features))
    return feat

# ----------------------------------------------------------------------------------------------------