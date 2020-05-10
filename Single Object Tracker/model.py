"""Model for the single object tracker."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras import layers
from tensorflow.keras import Model

# Rescale bounding box to fixed size  X

# Run through convnet and obtain feature vector

# Calculate distance between two vectors

# Create loss function training for distance

# Make training loop for the model


def rescale_bb(bounding_box, size):
    """Rescale boudning box to fixed size."""
    img = Image.fromarray(bouding_box)
    new_img = img.resize(size) if isinstance(size, tuple) else img.resize((size, size))
    return new_img


def calc_distance(vector1, vector2):
    """Calculate the distance between two feature vectors."""
    return np.linalg.norm(vector1 - vector2)


class TrackNet(Model):
    def __init__(self):
        super(TrackNet, self).__init__()
        self.conv1 = layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), 
            padding='valid', use_bias=False)
        self.norm1 = layers.BatchNormalization(axis=3)
        self.relu1 = layers.Activation('relu')
        
        self.conv2 = layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), 
            padding='valid', use_bias=False)
        self.norm2 = layers.BatchNormalization(axis=3)
        self.relu2 = layers.Activation('relu')
        self.pool2 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        self.drop2 = layers.Dropout(0.3)

        self.conv3 = layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), 
            padding='valid', use_bias=False)
        self.norm3 = layers.BatchNormalization(axis=3)
        self.relu3 = layers.Activation('relu')
        
        self.conv4 = layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), 
            padding='valid', use_bias=False)
        self.norm4 = layers.BatchNormalization(axis=3)
        self.relu4 = layers.Activation('relu')
        self.pool4 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        self.drop4 = layers.Dropout(0.3)

        self.conv5 = layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), 
            padding='valid', use_bias=False)
        self.norm5 = layers.BatchNormalization(axis=3)
        self.relu5 = layers.Activation('relu')
        self.pool5 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        self.drop5 = layers.Dropout(0.3)
        
        self.conv6 = layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1), 
            padding='valid', use_bias=False)
        self.norm6 = layers.BatchNormalization(axis=3)
        self.relu6 = layers.Activation('relu')
        self.flat6 = layers.Flatten()
        self.dens6 = layers.Dense(128, activation=None)
        self.l2norm = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=3)) # L2 normalize embeddings


    def call(self, input):
        x = self.conv1(input)
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
model = TrackNet()

# Define the loss
loss_object = tfa.losses.TripletSemiHardLoss()

# Define the optimizer
optimizer = tf.keras.optimizers.Adam()

# Define the metric
train_loss = tf.keras.metrics.Mean()


def train_model(epochs, batch_size):
    """Create training loop for the model."""
    print('Training the model for {} epochs...'.format(epochs))
    
    # Create empty list for the metrics
    train_loss_results = []

    # Training loop
    for epoch in range(epochs):
        for idx in range(0, dataset_size, batch_size):
            # Run over the batches in the dataset
            images = train_images[idx:min(idx+batch_size, dataset_size),:,:,:]
            labels = train_labels[idx:min(idx+batch_size, dataset_size),:,:,:]
            
            with tf.GradientTape() as tape:
                predictions = model(images, training=True)
                loss = loss_object(labels, predictions)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # Track progress
            train_loss.update_state(loss)

        # End of epoch
        if epoch % 5 == 0:
            print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(
                epoch, train_loss.result(), epoch_mean_iou.result()))

        train_loss_results.append(epoch_loss_avg.result())

train_model(epochs=10, batch_size=8)













