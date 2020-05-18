"""Train loop for the single object tracker."""

from __future__ import absolute_import, division, print_function

import pickle

import h5py
import matplotlib.pyplot as plt
import numpy as np

from data import get_combinations

# from model import TrackNetModel
# from utils import calc_distance, rescale_bb

# import tensorflow as tf


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

    return bounding_boxes, object_ids


def slice_image(im, labels_dict, seq, frame, id):
    dict_obj = labels_dict[seq]['frame'+str(frame)]['obj'+str(id)]
    left = dict_obj['left']
    top = dict_obj['top']
    right = dict_obj['right']
    bottom = dict_obj['bottom']

    im = im[top:bottom, left:right, :]

    return im


def run_validation(model):
    """Run validation sequence on model.

  Args:
    model: Model to train.
  """
    # Load the validation sequence

    # Create the initial embeddings

    # Run the model on the next frame

    # Link the embeddings to the embeddings of the previous frame

    # Update the metric

    # Return the MOT accuracy score


def train_model(model, epochs, batch_size, learning_rate):
    """Create training loop for the object tracker model.

  Args:
    model: Model to train.
    epochs: Number of epochs to train for.
    batch_size: Number of images per batch, must be odd number.
    learning_rate: Learning rate of the optimizer
  """
    # assert batch_size % 2 == 1, 'Batch size must be odd number'
    # print('Training the model for {} epochs...'.format(epochs))

    # # Define the loss and optimizer
    # loss_object = tfa.losses.TripletSemiHardLoss()
    # optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    #
    # # Define the metrics
    # train_loss = tf.keras.metrics.Mean()
    #
    # # Create empty list for the metrics
    # train_loss_results = []
    # mot_metric_results = []

    labels_file = '../data/kitti_first_seq_labels.bin'

    with open(labels_file, 'rb') as file:
        labels_dict = pickle.load(file)

    # Training loop
    for epoch in range(epochs):
        combinations = get_combinations(labels_file)

        for combination in combinations:
            data = h5py.File('../data/kitti_first_seq_images.h5', 'r')
            seq, id, anchor, positive, id_compare, negative = combination

            im_anchor = data[seq][anchor].copy()
            im_positive = data[seq][positive].copy()
            im_negative = data[seq][negative].copy()

            im_anchor = slice_image(im=im_anchor, labels_dict=labels_dict,
                                    seq=seq, frame=anchor, id=id)
            im_positive = slice_image(im=im_positive, labels_dict=labels_dict,
                                      seq=seq, frame=positive, id=id)
            im_negative = slice_image(im=im_negative, labels_dict=labels_dict,
                                      seq=seq, frame=negative, id=id_compare)

            print(im_anchor.shape)
            print(im_positive.shape)
            print(im_negative.shape)

            print(im_anchor)

            # Convert the [0.0 -  255.0] floats into int.
            im_anchor = np.array(im_anchor, dtype=int)
            im_positive = np.array(im_positive, dtype=int)
            im_negative = np.array(im_negative, dtype=int)
            print(im_anchor)

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
            ax1.imshow(im_anchor)
            ax2.imshow(im_positive)
            ax3.imshow(im_negative)
            plt.show()

            exit()

        # for video in range(len(dataset)):
        #     for obj in range(objects):
        #         # Get batch with anchor, positive and negative samples
        #         bounding_boxes, object_ids = get_batch(video, obj, batch_size)
        #
        #         with tf.GradientTape() as tape:
        #             embeddings = model(bounding_boxes, training=True)
        #             loss = loss_object(object_ids, embeddings)
        #
        #         gradients = tape.gradient(loss, model.trainable_variables)
        #         optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        #
        #         # Track progress
        #         train_loss.update_state(loss)

        # Run validation program on sequence and get score
        MOTA_score = run_validation(model)  # TO DO: implement this function

        if epoch % 2 == 0:
            print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(
                epoch, train_loss.result(), MOTA_score))

        # Append the results
        train_loss_results.append(train_loss.result())
        mot_metric_results.append(MOTA_score)

    # Visualize the results of training
    fig, axes = plt.subplots(2, sharex=True, figsize=(9, 6))
    fig.suptitle("Training Metrics", fontsize=14)

    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].plot(train_loss_results)

    axes[1].set_ylabel("Accuracy", fontsize=12)
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].plot(mot_metric_results)
    plt.show()


if __name__ == "__main__":
    # Main function
    # train_model(TrackNetModel, epochs=1, batch_size=3, learning_rate=0.001)
    train_model(epochs=1)
