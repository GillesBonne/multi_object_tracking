"""Model for the single object tracker."""

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.keras import Model, layers


class TrackNetV2(Model):
    def __init__(self, use_bias, l2_reg, use_dropout):
        super(TrackNetV2, self).__init__()
        self.use_dropout = use_dropout
        self.conv1 = layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
                                   padding='same', use_bias=use_bias,
                                   kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.relu1 = layers.Activation('relu')

        self.conv2 = layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
                                   padding='same', use_bias=use_bias,
                                   kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.relu2 = layers.Activation('relu')
        self.pool2 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        self.drop2 = layers.Dropout(0.3)

        self.conv3 = layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
                                   padding='same', use_bias=use_bias,
                                   kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.relu3 = layers.Activation('relu')

        self.conv4 = layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
                                   padding='same', use_bias=use_bias,
                                   kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.relu4 = layers.Activation('relu')
        self.pool4 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        self.drop4 = layers.Dropout(0.3)

        self.conv5 = layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
                                   padding='same', use_bias=use_bias,
                                   kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.relu5 = layers.Activation('relu')

        self.conv6 = layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
                                   padding='same', use_bias=use_bias,
                                   kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.relu6 = layers.Activation('relu')
        self.pool6 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        self.drop6 = layers.Dropout(0.3)

        self.conv7 = layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
                                   padding='same', use_bias=use_bias,
                                   kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.relu7 = layers.Activation('relu')

        self.conv8 = layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
                                   padding='same', use_bias=use_bias,
                                   kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.relu8 = layers.Activation('relu')
        self.pool8 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        self.drop8 = layers.Dropout(0.3)

        self.conv9 = layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
                                   padding='same', use_bias=use_bias,
                                   kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.relu9 = layers.Activation('relu')

        self.conv10 = layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
                                   padding='same', use_bias=use_bias,
                                   kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.relu10 = layers.Activation('relu')
        self.pool10 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        self.drop10 = layers.Dropout(0.3)

        self.conv11 = layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
                                   padding='same', use_bias=use_bias,
                                   kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.relu11 = layers.Activation('relu')

        self.conv12 = layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
                                   padding='same', use_bias=use_bias,
                                   kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.relu12 = layers.Activation('relu')
        self.pool12 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        self.drop12 = layers.Dropout(0.3)

        self.conv13 = layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
                                   padding='same', use_bias=use_bias,
                                   kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.relu13 = layers.Activation('relu')
        self.flat13 = layers.Flatten()
        self.dens13 = layers.Dense(128, activation=None)

        # L2 normalize embeddings.
        self.l2norm = layers.Lambda(lambda x: tf.math.l2_normalize(
            x, axis=1))

    def call(self, inputs):
        if len(inputs.shape) == 3:
            inputs = tf.expand_dims(inputs, 0)

        x = tf.cast(inputs, dtype=tf.float32)
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.drop2(x) if self.use_dropout else x

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        x = self.drop4(x) if self.use_dropout else x

        x = self.conv5(x)
        x = self.relu5(x)

        x = self.conv6(x)
        x = self.relu6(x)
        x = self.pool6(x)
        x = self.drop6(x) if self.use_dropout else x

        x = self.conv7(x)
        x = self.relu7(x)

        x = self.conv8(x)
        x = self.relu8(x)
        x = self.pool8(x)
        x = self.drop8(x) if self.use_dropout else x

        x = self.conv9(x)
        x = self.relu9(x)

        x = self.conv10(x)
        x = self.relu10(x)
        x = self.pool10(x)
        x = self.drop10(x) if self.use_dropout else x

        x = self.conv11(x)
        x = self.relu11(x)

        x = self.conv12(x)
        x = self.relu12(x)
        x = self.pool12(x)
        x = self.drop12(x) if self.use_dropout else x

        x = self.conv13(x)
        x = self.relu13(x)
        x = self.flat13(x)
        x = self.dens13(x)
        return self.l2norm(x)
