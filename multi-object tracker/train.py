"""Train loop for the single object tracker."""

from __future__ import absolute_import, division, print_function

import os
import pickle

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from data import get_combinations
from embeds import EmbeddingsDatabase
from eval import MOTMetric
from model import TrackNet
from utils import (check_acceptable_splits, resize_bb, show_frame_with_bb,
                   slice_image)


def get_batch(images_file, labels_file, combination, image_size=128):
    """Get batch of bouding box data from the dataset.

  Args:
    image_file: File that contains the images.
    label_file: File that contains the label information.
    combination: Tuple with the bounding box combinations to make.
    image_size: Size of the output bouding boxes.

  Returns:
    Both resized bounding box images and there corresponding ids.
  """
    # Batch always contains triplets.
    batch_size = 3

    seq, pos_id, anc_frame, pos_frame, neg_id, neg_frame = combination
    image_array = np.empty([batch_size, image_size, image_size, 3], dtype=np.uint8)

    # Get the frame data.
    with h5py.File(images_file, 'r') as data:
        im_anchor = data[seq][anc_frame].copy()
        im_positive = data[seq][pos_frame].copy()
        im_negative = data[seq][neg_frame].copy()

    # Get the label information.
    with open(labels_file, 'rb') as file:
        labels_dict = pickle.load(file)

    dict_anchor = labels_dict[seq]['frame'+str(anc_frame)]['obj'+str(pos_id)]
    dict_positive = labels_dict[seq]['frame'+str(pos_frame)]['obj'+str(pos_id)]
    dict_negative = labels_dict[seq]['frame'+str(neg_frame)]['obj'+str(neg_id)]

    # Get the bounding box of every object.
    anchor_bb = slice_image(im=im_anchor, dict_obj=dict_anchor)
    positive_bb = slice_image(im=im_positive, dict_obj=dict_positive)
    negative_bb = slice_image(im=im_negative, dict_obj=dict_negative)

    # Rescale the bouding boxes to fixed size.
    image_array[0, :, :, :] = resize_bb(anchor_bb, image_size)
    image_array[1, :, :, :] = resize_bb(positive_bb, image_size)
    image_array[2, :, :, :] = resize_bb(negative_bb, image_size)

    label_array = np.array([pos_id, pos_id, neg_id])

    return image_array, label_array


def run_validation(model, images_file, labels_file, sequences_val, image_size=128, visual=False):
    """Run validation sequence on model.

  Args:
    model: Model on which to perform the validation.
    image_file: Video sequence used for the validation.
    label_file: Corresponding label file.
    image_size: Size of the bounding boxes after resize.
    visual: Visualize the frame with bounding boxes and ids.
  """
    mot_validation = MOTMetric(auto_id=True)
    embeds_database = EmbeddingsDatabase(memory_length=15, memory_update=0.75)

    # Get the label file.
    with open(labels_file, 'rb') as file:
        label_dict = pickle.load(file)

    # Open the validation sequence.
    with h5py.File(images_file, 'r') as sequence:

        for seq in sequences_val:
            seq_name = 'seq'+str(seq)

            # Create the initial embeddings.
            init_frame = sequence[seq_name][0].copy()
            init_labels = label_dict[seq_name]['frame0']

            new_embeddings, embeds_dict = [], {}
            for object_dict in init_labels.values():
                # Get the bounding box of every object.
                object_bb = slice_image(im=init_frame, dict_obj=object_dict)
                object_bb = resize_bb(object_bb, image_size)
                embedding = model(object_bb)

                new_embeddings.append(embedding)

            # Perform the re-identification
            hypothesis_ids = embeds_database.match_embeddings(new_embeddings, max_distance=0.2)

            # Loop over every frame in the sequence (starting at second frame).
            for i, frame in enumerate(sequence[seq_name][1:]):
                curr_label = label_dict[seq_name]['frame'+str(i+1)]

                new_embeddings, object_ids = [], []
                object_bbs = np.empty((0, 4), dtype=int)
                for object_dict in curr_label.values():
                    # Get the bounding box of every object.
                    object_bb = slice_image(im=frame, dict_obj=object_dict)
                    object_bb = resize_bb(object_bb, image_size)
                    embedding = model(object_bb)

                    new_embeddings.append(embedding)
                    object_ids.append(object_dict['track_id'])

                    # Append the bounding box data.
                    left = object_dict['left']
                    top = object_dict['top']
                    right = object_dict['right']
                    bottom = object_dict['bottom']

                    object_bbs = np.append(object_bbs, np.array(
                        [[left, top, right, bottom]]), axis=0)

                # Perform the re-identification
                hypothesis_ids = embeds_database.match_embeddings(new_embeddings, max_distance=0.2)

                # Update the MOT metric.
                hypothese_bbs = object_bbs.copy()  # NOTE: THIS IS TEMPORARY!
                mot_validation.update(object_ids, hypothesis_ids,
                                      object_bbs.copy(), hypothese_bbs.copy())

                if visual:
                    # Visualize the frame with bouding boxes and ids.
                    show_frame_with_bb(frame, object_bbs.copy(), hypothesis_ids)

        # Return the MOT validation object
        return mot_validation


def train_model(model, image_files, label_files, epochs, learning_rate,
                sequences_train, sequences_val):
    """Create training loop for the object tracker model.

  Args:
    model: Model to train
    image_files: List of file names for the video sequences
    label_files: List of file names for the video labels
    epochs: Number of epochs to train for
    learning_rate: Learning rate of the optimizer
  """
    print('Training the model for {} epochs...'.format(epochs))

    # Create empty list for the metrics.
    train_loss_results = []
    mot_metric_results = []

    # Define the loss, optimizer and metric(s).
    loss_object = tfa.losses.TripletSemiHardLoss()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    train_loss = tf.keras.metrics.Mean()

    # Training loop.
    for epoch in range(epochs):
        # Get all bouding box combinations for this sequence.
        combinations = get_combinations(labels_file, sequences_train)

        for combination in combinations:
            images, labels = get_batch(images_file, labels_file, combination)

            with tf.GradientTape() as tape:
                embeddings = model(images, training=True)
                loss = loss_object(labels, embeddings)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # Track progress.
            train_loss.update_state(loss)

        # Run validation program on sequence and get score.
        MOTA_score = run_validation(model, images_file, labels_file, sequences_val)

        if epoch % 1 == 0:
            print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.1%}".format(
                epoch, train_loss.result(), MOTA_score.get_MOTA()))

        # Append the results.
        train_loss_results.append(train_loss.result())
        mot_metric_results.append(MOTA_score.get_MOTA())

    # Visualize the results of training.
    fig, axes = plt.subplots(2, sharex=True, figsize=(7, 5))
    fig.suptitle("Training Metrics", fontsize=14)

    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].plot(train_loss_results)

    axes[1].set_ylabel("Accuracy", fontsize=12)
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].plot(mot_metric_results)
    plt.show()

    return model


if __name__ == "__main__":
    # Select the model and data.
    model = TrackNet(padding='valid', use_bias=False)
    images_file = '../data/kitti_images.h5'
    labels_file = '../data/kitti_labels.bin'

    # Settings for the train process
    epochs = 3
    learning_rate = 0.001

    # Choose train/val/test.
    sequences_train = [0, 1]
    sequences_val = [2]
    sequences_test = [3]
    check_acceptable_splits('kitti', sequences_train, sequences_val, sequences_test)

    # Train the model
    model = train_model(model, images_file, labels_file,
                        epochs, learning_rate,
                        sequences_train, sequences_val)

    # Save the weights of the model
    model_path = "model/model.ckpt"
    model.save_weights(model_path)

    # Load the previously saved weights
    new_model = TrackNet(padding='valid', use_bias=False)
    new_model.load_weights(model_path)

    # Run the validation with visualization
    MOT_object = run_validation(new_model, images_file, labels_file, sequences_val, visual=True)
    print(MOT_object.get_MOTA())
