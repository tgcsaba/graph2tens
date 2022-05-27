import numpy as np
import tensorflow as tf

from tensorflow.keras import backend, regularizers

# ----------------------------------------------------------------------------------------------------

class Rank1TensorL2(regularizers.Regularizer):    
    def __init__(self, l2=0.01, **kwargs):
        self.l2 = backend.cast_to_floatx(l2)

    def __call__(self, x):
        return self.l2 * tf.reduce_sum(
            tf.math.cumprod(tf.reduce_sum(tf.square(x), axis=1), axis=0))

    def get_config(self):
        return {'l2': float(self.l2)}
    
# ----------------------------------------------------------------------------------------------------

class Rank1TensorL1(regularizers.Regularizer):
    def __init__(self, l1=0.01, **kwargs):
        self.l1 = backend.cast_to_floatx(l1)

    def __call__(self, x):
        return self.l1 * tf.reduce_sum(
            tf.math.cumprod(tf.reduce_sum(tf.abs(x), axis=1), axis=0))
        
# ----------------------------------------------------------------------------------------------------