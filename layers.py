from tensorflow import keras
from tensorflow.keras import backend as K
import numpy as np


class PerImageStandardization(keras.layers.Layer):
    def call(self, inputs, **kwargs):
        num_pixels = K.prod(K.shape(inputs)[-3:])
        min_stddev = 1 / K.sqrt(K.cast(num_pixels, inputs.dtype))
        adjusted_stddev = K.maximum(K.std(inputs, axis=(-1, -2, -3),
                                          keepdims=True), min_stddev)
        return (inputs - K.mean(inputs, axis=(-1, -2, -3), keepdims=True)) / adjusted_stddev

class NormalizingLayer01(keras.layers.Layer):
    
    def __init__(self, trainable=False, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.mean = K.constant([0., 0., 0.], dtype=K.floatx())
        self.std = K.constant([255., 255., 255.], dtype=K.floatx())

    def call(self, inputs, **kwargs):
        out = (inputs - self.mean) / self.std
        return out