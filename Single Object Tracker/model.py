"""Model for the single object tracker."""

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.keras import Model, layers


class TrackNet(Model):
    def __init__(self, padding, use_bias, data_format):
        super(TrackNet, self).__init__()
        axis = 1 if data_format == 'channel_first' else 3

        self.conv1 = layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
                                   padding=padding, use_bias=use_bias)
        self.norm1 = layers.BatchNormalization(axis=axis)
        self.relu1 = layers.Activation('relu')

        self.conv2 = layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
                                   padding=padding, use_bias=use_bias)
        self.norm2 = layers.BatchNormalization(axis=axis)
        self.relu2 = layers.Activation('relu')
        self.pool2 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        self.drop2 = layers.Dropout(0.3)

        self.conv3 = layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
                                   padding=padding, use_bias=use_bias)
        self.norm3 = layers.BatchNormalization(axis=axis)
        self.relu3 = layers.Activation('relu')

        self.conv4 = layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
                                   padding=padding, use_bias=use_bias)
        self.norm4 = layers.BatchNormalization(axis=axis)
        self.relu4 = layers.Activation('relu')
        self.pool4 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        self.drop4 = layers.Dropout(0.3)

        self.conv5 = layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
                                   padding=padding, use_bias=use_bias)
        self.norm5 = layers.BatchNormalization(axis=axis)
        self.relu5 = layers.Activation('relu')
        self.pool5 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        self.drop5 = layers.Dropout(0.3)

        self.conv6 = layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1),
                                   padding=padding, use_bias=use_bias)
        self.norm6 = layers.BatchNormalization(axis=axis)
        self.relu6 = layers.Activation('relu')
        self.flat6 = layers.Flatten()
        self.dens6 = layers.Dense(128, activation=None)
        self.l2norm = layers.Lambda(lambda x: tf.math.l2_normalize(
            x, axis=1))  # L2 normalize embeddings

    def call(self, inputs):
        if len(inputs.shape) == 3:
            inputs = tf.expand_dims(inputs, 0)

        x = tf.cast(inputs, dtype=tf.float32)
        x = self.conv1(x)
        x = self.relu1(self.norm1(x))

        x = self.conv2(x)
        x = self.relu1(self.norm2(x))
        x = self.pool2(x)
        x = self.drop2(x)

        x = self.conv3(x)
        x = self.relu3(self.norm3(x))

        x = self.conv4(x)
        x = self.relu4(self.norm4(x))
        x = self.pool4(x)
        x = self.drop4(x)

        x = self.conv5(x)
        x = self.relu5(self.norm5(x))
        x = self.pool5(x)
        x = self.drop5(x)

        x = self.conv6(x)
        x = self.relu6(self.norm6(x))
        x = self.flat6(x)
        x = self.dens6(x)
        return self.l2norm(x)


# Create an instance of the model
TrackNetModel = TrackNet(padding='valid', use_bias=False, data_format='channel_last')
