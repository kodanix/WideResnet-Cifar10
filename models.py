from abc import ABC, abstractmethod
from tensorflow import keras
#import tensorflow_addons as tfa
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.layers import add
from tensorflow.keras.layers import BatchNormalization
from layers import PerImageStandardization, NormalizingLayer01


class WideResidualNetwork():
    
    def __init__(self, depth=28, width=2, dropout_rate=0.0,
                 preprocess_mode=None,
                 out_activation=None) -> None:
        if (depth - 4) % 6 != 0:
            raise ValueError('Depth of the network must be such that (depth - 4)'
                             'should be divisible by 6.')
        self.depth = depth
        self.width = width
        self.dropout_rate = dropout_rate
        self.out_activation = out_activation
        self.initializer = "he_normal"
        self.preprocess_mode = preprocess_mode

    def get_name(self):
        return "wide-resnet-" + str(self.depth) + "-" + str(self.width) + "-" + self.initializer

    def build_model(self, input_shape, nb_classes):
        inp = Input(input_shape)
        if self.preprocess_mode == 'PIS':
            img_input = PerImageStandardization()(inp)
        elif self.preprocess_mode == 'FS':
            img_input = NormalizingLayer01()(inp)
            # img_input = PreProcessingLayer()(inp)
        elif self.preprocess_mode is None:
            img_input = inp
        else:
            raise Exception('Not supported: {0}'.format(self.preprocess_mode))

        x = self.__create_wide_residual_network(nb_classes, img_input, self.depth, self.width,
                                                self.dropout_rate, self.out_activation)
        # Create model.
        model = Model(inp, x, name='wide-resnet')
        return model

    def make_conv(self, n_kernels, kernel_size, padding='same', strides=1, activation='linear', use_bias=True):
        return Conv2D(n_kernels,
                      kernel_size,
                      padding=padding,
                      strides=strides,
                      kernel_initializer=self.initializer,
                      activation=activation, use_bias=use_bias)

    def block(self, x, n_filters, stride):
        o1 = Activation('relu')(BatchNormalization()(x))
        y = self.make_conv(n_filters, (3, 3), strides=stride, padding='same', use_bias=False)(o1)
        o2 = Activation('relu')(BatchNormalization()(y))
        z = self.make_conv(n_filters, (3, 3), strides=1, padding='same', use_bias=False)(o2)
        if x.shape[-1] != n_filters:
            return add([z, self.make_conv(n_filters, (1, 1), strides=stride, padding='same', use_bias=False)(o1)])
        else:
            return add([z, x])

    def group(self, o, n_filters, stride, n):
        for i in range(n):
            o = self.block(o, n_filters, stride=stride if i == 0 else 1)
        return o

    def __create_wide_residual_network(self, nb_classes, img_input, depth=28,
                                       width=8, dropout=0.0, activation=None):
        N = (depth - 4) // 6

        x = self.make_conv(16, (3, 3), padding='same', use_bias=False)(img_input)
        g0 = self.group(x, 16 * width, stride=1, n=N)
        g1 = self.group(g0, 32 * width, stride=2, n=N)
        g2 = self.group(g1, 64 * width, stride=2, n=N)

        x = Activation('relu')(BatchNormalization()(g2))
        x = GlobalAveragePooling2D()(x)
        x = Dense(nb_classes, activation=activation, kernel_initializer=self.initializer)(x)

        return x