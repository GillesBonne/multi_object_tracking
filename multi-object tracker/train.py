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
from model_v2 import TrackNetV2
from utils import (check_acceptable_splits, export_parameters, get_embeddings,
                   load_overfit_bboxes, resize_bb, show_frame_with_ids,
                   show_frame_with_labels, show_overfit_statistics,
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
    seq, pos_id, anc_frame, pos_frame, neg_id, neg_frame = combination

    # Batch always contains triplets.
    image_array = np.empty([3, image_size, image_size, 3], dtype=np.uint8)

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

    # Make sure that the IDs are unique between sequences.
    seq_num = ''.join(filter(str.isdigit, seq))
    pos_id = int(seq_num+str(pos_id))
    neg_id = int(seq_num+str(neg_id))

    label_array = np.array([pos_id, pos_id, neg_id])

    return image_array, label_array


def run_validation(model, settings, image_size=128, visual=None, visual_location=None):
    """Run validation sequence on model.

  Args:
    model: Model on which to perform the validation.
    image_file: Video sequence used for the validation.
    label_file: Corresponding label file.
    image_size: Size of the bounding boxes after resize.
    visual: Visualize the frame with bounding boxes and ids.
  """
    mot_metric = MOTMetric(auto_id=True)
    embeds_database = EmbeddingsDatabase(settings.memory_length, settings.memory_update)

    # Get the label file.
    with open(settings.labels_file, 'rb') as file:
        labels_dict = pickle.load(file)

    # Open the validation sequence.
    with h5py.File(settings.images_file, 'r') as sequence:
        # Loop over every validation sequence
        for seq in settings.sequences_val:
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
                if settings.detector:
                    embeddings, boxes, labels, probs = model(frame)
                    hyp_bbs = np.array(boxes, dtype=int)
                else:
                    embeddings = get_embeddings(model, frame, gt_labels)
                    hyp_bbs = obj_bbs.copy()

                # Perform the re-identification
                hyp_ids = embeds_database.match_embeddings(embeddings, settings.max_distance)

                # Update the MOT metric.
                mot_metric.update(obj_ids, hyp_ids,
                                  np.array(obj_bbs.copy()), np.array(hyp_bbs.copy()))  # << CHANGE THIS BACK!

                if visual == 're-id':
                    # Visualize the frame with bouding boxes and ids.
                    show_frame_with_ids(frame, hyp_bbs.copy(), hyp_ids,
                                        frame_num=i, seq_name='seq{}'.format(str(seq)),
                                        visual_location=visual_location)
                elif visual == 'detect':
                    show_frame_with_labels(frame, boxes, labels, probs)

        # Return the MOT metric object
        return mot_metric, embeds_database.get_average_cost()


def train_model(model, settings, save_directory):
    """Create training loop for the object tracker model.

  Args:
    model: Model to train
    image_files: List of file names for the video sequences
    label_files: List of file names for the video labels
    epochs: Number of epochs to train for
    learning_rate: Learning rate of the optimizer
  """
    # Create empty list for the metrics.
    train_loss_results = []
    mot_metric_epochs = []
    mot_accuracy_results = []
    mot_switches_results = []
    avg_cost_results = []

    # Get metric data before training.
    if settings.detector:
        tracker = MultiTrackNet(model)
        MOT_metric, avg_cost = run_validation(tracker, settings)
    else:
        MOT_metric, avg_cost = run_validation(model, settings)

    print('\nTraining the model for {} epochs...'.format(settings.epochs))

    print('\nAmount of combinations per epoch: {}.'.format(len(get_combinations(
        settings.labels_file, settings.sequences_train, settings.window_size,
        settings.num_combi_per_obj_per_epoch))))

    print("\nEpoch  -1: Acc:{:.1%}, Avg embed cost:{:.3f}, Switches:{}".format(
        MOT_metric.get_MOTA(), avg_cost, MOT_metric.get_num_switches()))

    mot_metric_epochs.append(-1)
    mot_accuracy_results.append(MOT_metric.get_MOTA())
    mot_switches_results.append(MOT_metric.get_num_switches())
    avg_cost_results.append(avg_cost)

    # Define the loss, optimizer and metric(s).
    loss_object = tfa.losses.TripletSemiHardLoss(margin=2.0)
    optimizer = tf.keras.optimizers.Adam(learning_rate=settings.learning_rate)
    train_loss = tf.keras.metrics.Mean()

    # Load the overfit bounding boxes and show them.
    # bboxes = load_overfit_bboxes()
    # show_overfit_statistics(model, bboxes)

    # Create empty array for image and label batch.
    triplet_batch = settings.triplet_batch
    image_batch = np.empty((triplet_batch*3, 128, 128, 3), dtype=np.uint8)
    label_batch = np.empty((triplet_batch*3), dtype=np.float32)

    # Training loop.
    for epoch in range(settings.epochs):
        # Get all bouding box combinations for this sequence.
        combinations = get_combinations(settings.labels_file, settings.sequences_train,
                                        settings.window_size, settings.num_combi_per_obj_per_epoch)

        length = len(combinations)
        for idx in range(0, length, triplet_batch):
            batch = combinations[idx:min(idx+triplet_batch, length)]

            for i, comb in enumerate(batch):
                images, labels = get_batch(settings.images_file, settings.labels_file, comb)
                image_batch[i*3:i*3+3, :, :, :] = images
                label_batch[i*3:i*3+3] = labels

            with tf.GradientTape() as tape:
                embeddings = model(image_batch, training=True)
                loss = loss_object(y_true=label_batch, y_pred=embeddings)
                loss += sum(model.losses)  # Add the L2 regularization loss

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # Track progress.
            train_loss.update_state(loss)

        # Show statistics of the training process.
        print("Epoch {:03d}: Loss:{:.3f}".format(epoch, train_loss.result()))

        # show_overfit_statistics(model, bboxes)

        # Append the results.
        train_loss_results.append(train_loss.result())

        if epoch % settings.val_epochs == 0:
            # Run validation program on sequence and get score.
            if settings.detector:
                tracker = MultiTrackNet(model)
                MOT_metric, avg_cost = run_validation(tracker, settings)
            else:
                MOT_metric, avg_cost = run_validation(model, settings)

            # Print statistics with accuracy and precision.
            print("Epoch {:03d}: Loss:{:.3f}, Acc:{:.1%}, Avg embed cost:{:.3f}, Switches:{}".format(
                epoch, train_loss.result(), MOT_metric.get_MOTP(), avg_cost, MOT_metric.get_num_switches()))

            # Append the results.
            mot_metric_epochs.append(epoch)
            mot_accuracy_results.append(MOT_metric.get_MOTA())
            mot_switches_results.append(MOT_metric.get_num_switches())
            avg_cost_results.append(avg_cost)

    print('\nTraining completed, exporting results.')

    # Visualize the results of training.
    fig, axes = plt.subplots(4, sharex=True, figsize=(14, 10))

    axes[0].set_ylabel("Loss")
    axes[0].plot(train_loss_results)

    axes[1].set_ylabel("Accuracy")
    axes[1].plot(mot_metric_epochs, mot_accuracy_results)

    axes[2].set_ylabel("Number of switches")
    axes[2].plot(mot_metric_epochs, mot_switches_results)

    axes[3].set_ylabel("Average cost")
    axes[3].set_xlabel("Epoch")
    axes[3].plot(mot_metric_epochs, avg_cost_results)

    fig.savefig(save_directory + '/metrics.png', bbox_inches='tight')
    plt.close()

    # Save training metrics.
    np.savetxt(save_directory + '/train_loss.txt', train_loss_results)
    np.savetxt(save_directory + '/mot_epochs.txt', mot_metric_epochs)
    np.savetxt(save_directory + '/mot_accuracy.txt', mot_accuracy_results)
    np.savetxt(save_directory + '/mot_switches.txt', mot_switches_results)
    np.savetxt(save_directory + '/avg_cost.txt', avg_cost_results)

    return model


class Settings:
    """Class for the settings of the train process."""
    epochs = 21
    learning_rate = 0.001  # Should be smaller or equal than 0.01.
    l2_reg = 0.0001
    use_dropout = False

    # Settings for the dataset and triplets.
    dataset = 'kitti'
    images_file = '../data/kitti_images.h5'
    labels_file = '../data/kitti_labels.bin'

    sequences_train = [12]
    sequences_val = [12]
    sequences_test = [12]
    allow_overfit = True

    window_size = 3
    num_combi_per_obj_per_epoch = 10
    triplet_batch = 8  # Number of triplets in one batch.

    # Settings for the validation run.
    detector = False
    memory_length = 30
    memory_update = 0.75
    max_distance = 0.5

    # Run validation every n epochs.
    val_epochs = 20


if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Settings for the train process.
    settings = Settings()

    # Create unique folder for every training session.
    now = datetime.datetime.now()
    save_directory = 'saved_models/saved_model_' + now.strftime('%Y-%m-%d_%H-%M-%S')
    Path(save_directory).mkdir(parents=True, exist_ok=True)

    # Select the model and data.
    model = TrackNet(use_bias=False, l2_reg=settings.l2_reg, use_dropout=settings.use_dropout)

    # Check if choosen split is acceptable.
    check_acceptable_splits(settings.dataset, settings.sequences_train, settings.sequences_val,
                            settings.sequences_test, allow_overfit=settings.allow_overfit)

    # Save training parameters.
    export_parameters(save_directory, settings)

    # Train the model.
    model = train_model(model, settings, save_directory)

    # Save the weights of the model.
    model_path = save_directory + '/saved_model.ckpt'
    model.save_weights(model_path)

    # Load the previously saved weights.
    new_model = TrackNet(use_bias=False, l2_reg=settings.l2_reg, use_dropout=settings.use_dropout)
    new_model.load_weights(model_path)

    if settings.detector:
        # Extend the re-identification model with detection.
        tracker = MultiTrackNet(new_model)

        # Run the validation with visualization.
        MOT_metric, avg_cost = run_validation(
            tracker, settings, visual='re-id', visual_location=save_directory)
    else:
        # Run the validation with visualization.
        MOT_metric, avg_cost = run_validation(
            new_model, settings, visual='re-id', visual_location=save_directory)

    # Print some of the statistics.
    print('\nTest results:')
    print('Multi-object tracking accuracy: {:.1%}'.format(MOT_metric.get_MOTA()))
    print('Multi-object tracking switches: {}'.format(MOT_metric.get_num_switches()))
    print('Multi-object tracking avg embed cost: {:.3f}'.format(avg_cost))
    print('Multi-object detection precision: {:.1%}'.format(MOT_metric.get_precision()))
    print('Multi-object detection recall: {:.1%}\n'.format(MOT_metric.get_recall()))
