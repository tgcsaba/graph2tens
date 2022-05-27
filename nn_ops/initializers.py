import numpy as np
import tensorflow as tf

from math import sqrt

from tensorflow.keras import initializers
        
# ----------------------------------------------------------------------------------------------------

class FactorialInit(initializers.Initializer):
    def __init__(self, n_levels):
        """
        Factorial initializer for embed_coeffs in GSN.
        """
        self.n_levels = n_levels

    def __call__(self, shape, dtype=None):
        n_levels = shape[0]
        assert n_levels == self.n_levels
        factorials = tf.math.cumprod(tf.range(1, n_levels+1, dtype=dtype))
        return tf.tile(
            1. / factorials[[slice(n_levels)] + (len(shape)-1) * [None]], (1,) + tuple(shape[1:]))

# ----------------------------------------------------------------------------------------------------

def _compute_target_variances(n_dim, n_features, n_levels):
    """
    Computes the target variances for initializing the components of the rank-1 tensor components
    according to Glorot init.
    """
    
    variances = [2. / (n_dim + n_features)] + [(tf.pow(float(n_dim), float(i)) + n_features)
                                               / (tf.pow(float(n_dim), float(i+1)) + n_features)
                                               for i in range(1, n_levels)]
    return variances
        
# ----------------------------------------------------------------------------------------------------

class GSNUniformInit(initializers.Initializer):
    def __init__(self, n_levels):
        """
        Uniform Glorot initializer for GSN.
        """
        self.n_levels = n_levels
    
    def __call__(self, shape, dtype=None):
        _, n_dim, n_features = shape
        variances = _compute_target_variances(n_dim, n_features, self.n_levels)
        kernels = []
        for i, variance in enumerate(variances):
            current_shape = (1, n_dim, n_features)
            limit = tf.cast(tf.sqrt(3.) * tf.sqrt(variance), dtype)
            kernel = tf.random.uniform(current_shape, minval=-limit, maxval=limit, dtype=dtype)
            kernels.append(kernel)
        kernel = tf.concat(kernels, axis=0)
        return kernel

# ----------------------------------------------------------------------------------------------------

class GSNNormalInit(initializers.Initializer):
    def __init__(self, n_levels):
        """
        Normal Glorot initializer for GSN.
        """
        self.n_levels = n_levels
        
    def __call__(self, shape, dtype=None):
        _, n_dim, n_features = shape
        variances = _compute_target_variances(n_dim, n_features, self.n_levels)
        kernels = []
        for i, variance in enumerate(variances):
            current_shape = (1, n_dim, n_features)
            stddev = tf.sqrt(variance)
            kernel = tf.random.normal(current_shape, stddev=stddev, dtype=dtype)  
            kernels.append(kernel)
        kernel = tf.concat(kernels, axis=0)
        return kernel

# ----------------------------------------------------------------------------------------------------