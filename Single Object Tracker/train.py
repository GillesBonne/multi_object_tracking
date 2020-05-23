"""Train loop for the single object tracker."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import h5py
import pickle

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt

from data import get_combinations
from model import TrackNetModel, TrackNet
from eval import MOTMetric
from utils import resize_bb, slice_image, re_identification, show_frame_with_bb


def get_batch(image_file, label_file, combination, image_size=128):
    """Get batch of bouding box data from the dataset.

  Args:
    image_file: File that contains the images.
    label_file: File that contains the label information.
    combination: Tuple with the bounding box combinations to make.
    image_size: Size of the output bouding boxes.

  Returns:
    Both resized bounding box images and there corresponding ids.
  """
    batch_size = 3  # Batch always contains triplets
    seq, pos_id, anc_frame, pos_frame, neg_id, neg_frame = combination
    image_array = np.empty([batch_size, image_size, image_size, 3], dtype=np.uint8)

    # Get the frame data
    with h5py.File(image_file, 'r') as data:
        im_anchor = data[seq][anc_frame].copy()
        im_positive = data[seq][pos_frame].copy()
        im_negative = data[seq][neg_frame].copy()

    # Get the label information
    with open(label_file, 'rb') as file:
        labels_dict = pickle.load(file)
        
    dict_anchor = labels_dict[seq]['frame'+str(anc_frame)]['obj'+str(pos_id)]
    dict_positive = labels_dict[seq]['frame'+str(pos_frame)]['obj'+str(pos_id)]
    dict_negative = labels_dict[seq]['frame'+str(neg_frame)]['obj'+str(neg_id)]

    # Get the bounding box of every object
    anchor_bb = slice_image(im=im_anchor, dict_obj=dict_anchor)
    positive_bb = slice_image(im=im_positive, dict_obj=dict_positive)
    negative_bb = slice_image(im=im_negative, dict_obj=dict_negative)

    # Rescale the bouding boxes to fixed size
    image_array[0,:,:,:] = resize_bb(anchor_bb, image_size)
    image_array[1,:,:,:] = resize_bb(positive_bb, image_size)
    image_array[2,:,:,:] = resize_bb(negative_bb, image_size)

    label_array = np.array([pos_id, pos_id, neg_id])

    # # Check if the data is correct
    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    # ax1.imshow(image_array[0,:,:,:])
    # ax2.imshow(image_array[1,:,:,:])
    # ax3.imshow(image_array[2,:,:,:])
    # plt.show()

    return image_array, label_array 


def run_validation(model, image_file, label_file, image_size=128, visual=False):
    """Run validation sequence on model.

  Args:
    model: Model on which to perform the validation.
    image_file: Video sequence used for the validation.
    label_file: Corresponding label file. 
    image_size: Size of the bounding boxes after resize.
    visual: Visualize the frame with bounding boxes and ids.
  """
    mot_validation = MOTMetric(auto_id=True)

    # Get the label file
    with open(label_file, 'rb') as file:
        label_dict = pickle.load(file)

    # Open the validation sequence
    with h5py.File(image_file, 'r') as sequence:
        
        # Create the initial embeddings
        init_frame = sequence['seq0'][0].copy()
        init_labels = label_dict['seq0']['frame0']
        
        new_embeddings, embeds_dict = [], {}
        for object_dict in init_labels.values():
            # Get the bounding box of every object
            object_bb = slice_image(im=init_frame, dict_obj=object_dict)
            object_bb = resize_bb(object_bb, image_size)
            embedding = model(object_bb)

            new_embeddings.append(embedding)

        # Perform the re-identification
        embeds_dict, hypothesis_ids = re_identification(embeds_dict, new_embeddings, 
            method='Euclidean', max_dist=1.0, update=True)

        # Loop over every frame in the sequence (starting at second frame)
        for i, frame in enumerate(sequence['seq0'][1:]):
            curr_label = label_dict['seq0']['frame'+str(i+1)]
            
            new_embeddings, object_ids = [], []
            object_bbs = np.empty((0,4), dtype=int)
            for object_dict in curr_label.values():
                # Get the bounding box of every object
                object_bb = slice_image(im=frame, dict_obj=object_dict)
                object_bb = resize_bb(object_bb, image_size)
                embedding = model(object_bb)

                new_embeddings.append(embedding)
                object_ids.append(object_dict['track_id'])

                # Append the bounding box data
                left = object_dict['left']
                top = object_dict['top']
                right = object_dict['right']
                bottom = object_dict['bottom']

                object_bbs = np.append(object_bbs, np.array([[left, top, right, bottom]]), axis=0)

            # Perform the re-identification
            embeds_dict, hypothesis_ids = re_identification(embeds_dict, new_embeddings, 
                method='Euclidean', max_dist=1.0, update=True)

            # Update the MOT metric
            hypothese_bbs = object_bbs.copy()  # NOTE: THIS IS TEMPORARY!
            mot_validation.update(object_ids, hypothesis_ids, object_bbs.copy(), hypothese_bbs.copy())

            if visual:
                # Visualize the frame with bouding boxes and ids
                show_frame_with_bb(frame, object_bbs.copy(), hypothesis_ids)

        # Return the MOT accuracy score
        return mot_validation.get_MOTA()


def train_model(model, image_files, label_files, epochs, learning_rate):
    """Create training loop for the object tracker model.

  Args:
    model: Model to train
    image_files: List of file names for the video sequences
    label_files: List of file names for the video labels
    epochs: Number of epochs to train for
    learning_rate: Learning rate of the optimizer
  """
    print('Training the model for {} epochs...'.format(epochs))

    # Create empty list for the metrics
    train_loss_results = []
    mot_metric_results = []

    # Define the loss, optimizer and metric(s)
    loss_object = tfa.losses.TripletSemiHardLoss()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    train_loss = tf.keras.metrics.Mean()

    # Training loop
    for epoch in range(epochs):
        for image_file, label_file in zip(image_files, label_files):
            # Get all bouding box combinations for this sequence
            combinations = get_combinations(label_file)
            
            for combination in combinations:
                images, labels = get_batch(image_file, label_file, combination)

                with tf.GradientTape() as tape:
                    embeddings = model(images, training=True)             
                    loss = loss_object(labels, embeddings)

                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                # Track progress
                train_loss.update_state(loss)

        # Run validation program on sequence and get score
        MOTA_score = run_validation(model, image_files[0], label_files[0])

        if epoch % 5 == 0:
            print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.1%}".format(
                epoch, train_loss.result(), MOTA_score))

        # Append the results
        train_loss_results.append(train_loss.result())
        mot_metric_results.append(MOTA_score)

    # Visualize the results of training
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
    # Select the model and data
    model = TrackNetModel
    image_files = ['../data/kitti_first_seq_images.h5']
    label_files = ['../data/kitti_first_seq_labels.bin']
    
    # Settings for the train process
    epochs = 100
    learning_rate = 0.001

    # Train the model
    model = train_model(model, image_files, label_files, epochs, learning_rate)

    # Save the weights of the model
    model_path = "model/model.ckpt"
    model.save_weights(model_path)

    # Load the previously saved weights
    new_model = TrackNet(padding='valid', use_bias=False, data_format='channel_last')
    new_model.load_weights(model_path)

    # Run the validation with visualization
    run_validation(new_model, image_files[0], label_files[0], visual=True)





