"""Train loop for the single object tracker."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from model import TrackNetModel
from utils import rescale_bb, calc_distance


def get_batch(video, obj, batch_size):
    """Get batch of bouding box data from the dataset.

  Args:
    video: Current video number.
    obj: Current object id for the anchor bouding box. 
    batch_size: Number of samples in the batch, with anchor and
      number of positive and negative samples.

  Returns:
    The labels and bouding boxes of this batch. Labels must 
    contain ids for the bouding boxes where the anchor and the 
    positive samples must have the same id and the negative 
    samples should have different ids. 
  """
    # TO DO: LOAD BATCHES
    # NOTE: MAKE SURE THE BOUDING BOXES HAVE THE SAME (SQUARE) SIZE
    bounding_boxes = [rescale_bb(bb, size) for bb in bounding_boxes]

    return bounding_boxes, labels


def train_model(model, epochs, batch_size, learning_rate):
    """Create training loop for the object tracker model.

  Args:
    model: Model to train.
    epochs: Number of epochs to train for.
    batch_size: Number of images per batch, must be odd number.
    learning_rate: Learning rate of the optimizer
    """
    assert batch_size % 2 == 1, 'Batch size must be odd number'
    print('Training the model for {} epochs...'.format(epochs))

    # Define the loss and optimizer
    loss_object = tfa.losses.TripletSemiHardLoss()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Define the metrics
    train_loss = tf.keras.metrics.Mean()    

    # Create empty list for the metrics
    train_loss_results = []

    # Training loop
    for epoch in range(epochs):
        for video in range(len(dataset)):
            for obj in range(objects):
                # Get batch with anchor, positive and negative samples
                bounding_boxes, labels = get_batch(video, obj, batch_size)

                with tf.GradientTape() as tape:
                    predictions = model(bounding_boxes, training=True)
                    loss = loss_object(labels, predictions)

                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                # Track progress
                train_loss.update_state(loss)

        # End of epoch
        if epoch % 2 == 0:
            print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(
                epoch, train_loss.result(), 0))  # TO DO: Create metric for accuracy

        train_loss_results.append(train_loss.result())

    return train_loss_results  # Or call function for visualization


if __name__ == "__main__":
    # Main function
    train_model(TrackNetModel, epochs=10, batch_size=3, learning_rate=0.001)






