"""Train loop for the single object tracker."""

from __future__ import absolute_import, division, print_function

import datetime
import pickle
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from data import get_combinations
from embeds import EmbeddingsDatabase
from eval import MOTMetric
from model import TrackNet
from model_extension import MultiTrackNet
from utils import (check_acceptable_splits, export_parameters, resize_bb,
                   show_frame_with_ids, show_frame_with_labels, slice_image)


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


def run_validation(model, images_file, labels_file, sequences_val, memory_length,
                   memory_update, max_distance, image_size=128, visual=None, visual_location=None):
    """Run validation sequence on model.

  Args:
    model: Model on which to perform the validation.
    image_file: Video sequence used for the validation.
    label_file: Corresponding label file.
    image_size: Size of the bounding boxes after resize.
    visual: Visualize the frame with bounding boxes and ids.
  """
    mot_metric = MOTMetric(auto_id=True)
    embeds_database = EmbeddingsDatabase(memory_length, memory_update)

    # Get the label file.
    with open(labels_file, 'rb') as file:
        labels_dict = pickle.load(file)

    # Open the validation sequence.
    with h5py.File(images_file, 'r') as sequence:
        # Loop over every validation sequence
        for seq in sequences_val:
            # Loop over every frame in the current sequence
            for i, frame in enumerate(sequence['seq'+str(seq)]):
                # Get the ground truth labels for the current frame
                gt_labels = labels_dict['seq'+str(seq)]['frame'+str(i)]

                obj_ids, obj_bbs = [], []
                for label in gt_labels.values():
                    obj_ids.append(label['track_id'])
                    obj_bbs.append([label['left'], label['top'],
                                    label['right'], label['bottom']])

                # Get the embeddings and bouding boxes by running the model
                embeddings, boxes, labels, probs = model(frame)
                hyp_bbs = np.array(boxes, dtype=int)

                # Perform the re-identification
                hyp_ids = embeds_database.match_embeddings(embeddings, max_distance)

                # Update the MOT metric.
                mot_metric.update(obj_ids, hyp_ids,
                                  np.array(obj_bbs.copy()), np.array(hyp_bbs.copy()))

                if visual == 're-id':
                    # Visualize the frame with bouding boxes and ids.
                    show_frame_with_ids(frame, hyp_bbs.copy(), hyp_ids,
                                        frame_num=i, seq_name='seq{}'.format(str(seq)),
                                        visual_location=visual_location)
                elif visual == 'detect':
                    show_frame_with_labels(frame, boxes, labels, probs)

        # Return the MOT metric object
        return mot_metric, embeds_database.get_average_cost()


def train_model(model, images_file, labels_file, epochs, learning_rate,
                window_size, num_combi_per_obj_per_epoch,
                memory_length, memory_update, max_distance,
                sequences_train, sequences_val, val_epochs,
                save_directory):
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
    mot_metric_epochs = []
    mot_accuracy_results = []
    mot_precision_results = []
    mot_switches_results = []

    # Define the loss, optimizer and metric(s).
    loss_object = tfa.losses.TripletSemiHardLoss()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    train_loss = tf.keras.metrics.Mean()

    # Training loop.
    for epoch in range(epochs):
        # Get all bouding box combinations for this sequence.
        combinations = get_combinations(labels_file, sequences_train,
                                        window_size, num_combi_per_obj_per_epoch)

        for combination in combinations:
            images, labels = get_batch(images_file, labels_file, combination)

            with tf.GradientTape() as tape:
                embeddings = model(images, training=True)
                loss = loss_object(labels, embeddings)
                loss += sum(model.losses)  # Add the L2 regularization loss

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # Track progress.
            train_loss.update_state(loss)

        if epoch % val_epochs == 0:
            # Run validation program on sequence and get score.
            tracker = MultiTrackNet(model)
            MOT_metric, avg_cost = run_validation(tracker, images_file, labels_file,
                                                  sequences_val, memory_length, memory_update, max_distance)

            # Print statistics with accuracy and precision
            print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.1%}, Precision: {:.1%}, Average embed cost: {:.3f}".format(
                epoch, train_loss.result(), MOT_metric.get_MOTA(), MOT_metric.get_MOTP(), avg_cost))

            # Append the results
            train_loss_results.append(train_loss.result())
            mot_metric_epochs.append(epoch)
            mot_accuracy_results.append(MOT_metric.get_MOTA())
            mot_precision_results.append(MOT_metric.get_MOTP())
            mot_switches_results.append(MOT_metric.get_num_switches())
        else:
            # Print statistics without accuracy and precision
            print("Epoch {:03d}: Loss: {:.3f}".format(epoch, train_loss.result()))

            # Append the results.
            train_loss_results.append(train_loss.result())

    # Visualize the results of training.
    fig, axes = plt.subplots(4, sharex=True, figsize=(7, 7))
    fig.suptitle("Training Metrics", fontsize=14)

    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].plot(train_loss_results)

    axes[1].set_ylabel("Accuracy", fontsize=12)
    axes[1].plot(mot_metric_epochs, mot_accuracy_results)

    axes[2].set_ylabel("Precision", fontsize=12)
    axes[2].plot(mot_metric_epochs, mot_precision_results)

    axes[3].set_ylabel("Number of switches", fontsize=12)
    axes[3].set_xlabel("Epoch", fontsize=12)
    axes[3].plot(mot_metric_epochs, mot_switches_results)
    plt.show()

    # Save training metrics.
    np.savetxt(save_directory + '/train_loss.txt', train_loss_results)
    np.savetxt(save_directory + '/mot_epochs.txt', mot_metric_epochs)
    np.savetxt(save_directory + '/mot_accuracy.txt', mot_accuracy_results)
    np.savetxt(save_directory + '/mot_precision.txt', mot_precision_results)
    np.savetxt(save_directory + '/mot_switches.txt', mot_switches_results)

    return model


if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Settings for the train process.
    epochs = 1
    learning_rate = 0.01
    l2_reg = 0.001  # L2 regularization
    l2_norm = True  # L2 normalization

    memory_length = 30
    memory_update = 0.75
    max_distance = 0.1

    window_size = 10
    num_combi_per_obj_per_epoch = 1

    # Create unique folder for every training session.
    now = datetime.datetime.now()
    save_directory = 'saved_models/saved_model_' + now.strftime('%Y-%m-%d_%H-%M-%S')
    Path(save_directory).mkdir(parents=True, exist_ok=True)

    # Select the model and data.
    model = TrackNet(padding='valid', use_bias=False, l2_reg=l2_reg, l2_norm=l2_norm)
    images_file = '../data/kitti_images.h5'
    labels_file = '../data/kitti_labels.bin'

    # Choose train/val/test.
    sequences_train = [12]
    sequences_val = [12]
    sequences_test = [12]
    check_acceptable_splits('kitti', sequences_train, sequences_val, sequences_test,
                            allow_overfit=True)

    # Save training parameters.
    export_parameters(save_directory, learning_rate, l2_reg, l2_norm,
                      memory_length, memory_update, max_distance,
                      window_size, num_combi_per_obj_per_epoch,
                      sequences_train, sequences_val, sequences_test)

    print('Amount of combinations per epoch: ', len(get_combinations(
        labels_file, sequences_train, window_size, num_combi_per_obj_per_epoch)))

    # Run validation every n epochs.
    val_epochs = 10

    # Train the model.
    model = train_model(model, images_file, labels_file,
                        epochs, learning_rate,
                        window_size, num_combi_per_obj_per_epoch,
                        memory_length, memory_update, max_distance,
                        sequences_train, sequences_val, val_epochs,
                        save_directory)

    # Save the weights of the model.
    model_path = save_directory + '/saved_model.ckpt'
    model.save_weights(model_path)

    # Load the previously saved weights.
    new_model = TrackNet(padding='valid', use_bias=False, l2_reg=l2_reg, l2_norm=l2_norm)
    new_model.load_weights(model_path)

    # Extend the re-identification model with detection.
    tracker = MultiTrackNet(new_model)

    # Run the validation with visualization.
    MOT_metric, avg_cost = run_validation(tracker, images_file, labels_file,
                                          sequences_test, memory_length, memory_update, max_distance=0.1,
                                          visual='re-id', visual_location=save_directory)

    # Print some of the statistics.
    print('\nTest results:')
    print('Multi-object tracking accuracy: {:.1%}'.format(MOT_metric.get_MOTA()))
    print('Multi-object tracking precision: {:.1%}'.format(MOT_metric.get_MOTP()))
    print('Multi-object tracking avg embed cost: {:.3f}'.format(avg_cost))
    print('Multi-object detection precision: {:.1%}'.format(MOT_metric.get_precision()))
    print('Multi-object detection recall: {:.1%}\n'.format(MOT_metric.get_recall()))
