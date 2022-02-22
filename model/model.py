from numpy import block
import tensorflow as tf
import tensorflow.keras as keras
from .layers import SConvBlock, TConvBlock, STConvBlock, OutputLayer, FullyConLayer


class ConvLSTM1D_Custom(keras.Model):
    def __init__(self, input_shape, graph_kernel, n_his, Ks, Kt, blocks, act_func, norm, dropout, **kwargs):
        super(ConvLSTM1D_Custom, self).__init__(name = "ConvLSTM1D_Custom" ,**kwargs)
        self.n_his = n_his
        self.n = input_shape[1]
        self.outc = input_shape[-1]
        assert len(blocks) == 1, "STGCN-A model allows only 1 group of Spatio-Conv layers & 1 group of Temporal-Conv layers"
        # Format - All spatio layers, followed by all temporal layers - blocks contains output units for each layer
        # self.sconv_block = SConvBlock(graph_kernel, Ks, input_shape[-1], blocks[0], norm, dropout)
        self.tconv_block = []
        for channels in blocks[0][:-1]:
            self.tconv_block.append(keras.layers.ConvLSTM1D(channels, 3, return_sequences=True, padding="same"))
        self.tconv_block.append(keras.layers.ConvLSTM1D(blocks[0][-1], 3, return_sequences=False, padding="same"))
        self.norm = norm
        self.dropout_layer = keras.layers.Dropout(rate = dropout)
        if norm == "batch":
            self.normalization = keras.layers.BatchNormalization(axis=[2,3])
        elif norm == "layer":
            self.normalization = keras.layers.LayerNormalization(axis=[2,3])
        elif norm != "L2":
            raise NotImplementedError(f'ERROR: Normalization function "{norm}" is not implemented.')
        # Output Layer
        self.layer3 = FullyConLayer(input_shape[1], blocks[0][-1], input_shape[-1])

    @tf.function
    def call(self, x:tf.Tensor):
        x = tf.cast(x, tf.float64)
        # x = self.sconv_block(x)
        for layer in self.tconv_block:
            x = layer(x)
        x = self.dropout_layer(x)
        x = tf.reshape(x, (-1, 1, *(x.shape[1:])))
        if self.norm == "L2":
            x = tf.nn.l2_normalize(x, axis=[2,3])
        else:
            x = self.normalization(x)
        y = self.layer3(x)
        return tf.reshape(y, shape=[-1, 1, self.n, self.outc])


class STGCN_ConvLSTM1D(keras.Model):
    def __init__(self, input_shape, graph_kernel, n_his, Ks, Kt, blocks, act_func, norm, dropout, **kwargs):
        super(STGCN_ConvLSTM1D, self).__init__(name = "STGCN_ConvLSTM1D" ,**kwargs)
        self.n_his = n_his
        self.n = input_shape[1]
        self.outc = input_shape[-1]
        assert len(blocks) == 2, "STGCN-A model allows only 1 group of Spatio-Conv layers & 1 group of Temporal-Conv layers"
        # Format - All spatio layers, followed by all temporal layers - blocks contains output units for each layer
        self.sconv_block = SConvBlock(graph_kernel, Ks, input_shape[-1], blocks[0], norm, dropout)
        self.tconv_block = []
        for channels in blocks[1][:-1]:
            self.tconv_block.append(keras.layers.ConvLSTM1D(channels, 3, return_sequences=True, padding="same"))
        self.tconv_block.append(keras.layers.ConvLSTM1D(blocks[1][-1], 3, return_sequences=False, padding="same"))
        self.norm = norm
        self.dropout_layer = keras.layers.Dropout(rate = dropout)
        if norm == "batch":
            self.normalization = keras.layers.BatchNormalization(axis=[2,3])
        elif norm == "layer":
            self.normalization = keras.layers.LayerNormalization(axis=[2,3])
        elif norm != "L2":
            raise NotImplementedError(f'ERROR: Normalization function "{norm}" is not implemented.')
        # Output Layer
        self.layer3 = FullyConLayer(input_shape[1], blocks[-1][-1], input_shape[-1])

    @tf.function
    def call(self, x:tf.Tensor):
        x = tf.cast(x, tf.float64)
        x = self.sconv_block(x)
        for layer in self.tconv_block:
            x = layer(x)
        x = self.dropout_layer(x)
        x = tf.reshape(x, (-1, 1, *(x.shape[1:])))
        if self.norm == "L2":
            x = tf.nn.l2_normalize(x, axis=[2,3])
        else:
            x = self.normalization(x)
        y = self.layer3(x)
        return tf.reshape(y, shape=[-1, 1, self.n, self.outc])


class Conv2D_ConvLSTM1D(keras.Model):
    def __init__(self, input_shape, graph_kernel, n_his, Ks, Kt, blocks, act_func, norm, dropout, **kwargs):
        super(Conv2D_ConvLSTM1D, self).__init__(name = "Conv2D_ConvLSTM1D" ,**kwargs)
        self.n_his = n_his
        self.n = input_shape[1]
        self.outc = input_shape[-1]
        assert len(blocks) == 2, "STGCN-A model allows only 1 group of Spatio-Conv layers & 1 group of Temporal-Conv layers"
        # Format - All spatio layers, followed by all temporal layers - blocks contains output units for each layer
        self.sconv_block = []
        for channels in blocks[0]:
            self.sconv_block.append(keras.layers.Conv2D(channels, 3, padding='same'))
        self.tconv_block = []
        for channels in blocks[1][:-1]:
            self.tconv_block.append(keras.layers.ConvLSTM1D(channels, 3, return_sequences=True, padding="same"))
        self.tconv_block.append(keras.layers.ConvLSTM1D(blocks[1][-1], 3, return_sequences=False, padding="same"))
        self.norm = norm
        self.dropout_layer1 = keras.layers.Dropout(rate = dropout)
        self.dropout_layer2 = keras.layers.Dropout(rate = dropout)
        if norm == "batch":
            self.normalization1 = keras.layers.BatchNormalization(axis=[2,3])
            self.normalization2 = keras.layers.BatchNormalization(axis=[2,3])
        elif norm == "layer":
            self.normalization1 = keras.layers.LayerNormalization(axis=[2,3])
            self.normalization2 = keras.layers.LayerNormalization(axis=[2,3])
        elif norm != "L2":
            raise NotImplementedError(f'ERROR: Normalization function "{norm}" is not implemented.')
        # Output Layer
        self.layer3 = FullyConLayer(input_shape[1], blocks[-1][-1], input_shape[-1])

    @tf.function
    def call(self, x:tf.Tensor):
        x = tf.cast(x, tf.float64)
        for layer in self.sconv_block:
            x = layer(x)
        x = self.dropout_layer1(x)
        if self.norm == "L2":
            x = tf.nn.l2_normalize(x, axis=[2,3])
        else:
            x = self.normalization1(x)
        for layer in self.tconv_block:
            x = layer(x)
        x = self.dropout_layer2(x)
        x = tf.reshape(x, (-1, 1, *(x.shape[1:])))
        if self.norm == "L2":
            x = tf.nn.l2_normalize(x, axis=[2,3])
        else:
            x = self.normalization2(x)
        y = self.layer3(x)
        return tf.reshape(y, shape=[-1, 1, self.n, self.outc])