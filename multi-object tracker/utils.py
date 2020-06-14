"""Utilary functions for the single object tracker."""

from __future__ import absolute_import, division, print_function

from pathlib import Path

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from yolo.utils.box import visualize_boxes

def get_embeddings(model, frame, label_dict, image_size=128):
    """Return the embedding of the model for given set of bounding boxes."""
    embeddings = []
    for label in label_dict.values():
        bbox = slice_image(frame.copy(), label)
        bbox = resize_bb(bbox, image_size)
        embeddings.append(model(bbox).numpy())

    return embeddings


def slice_image(im, dict_obj):
    """Slice the bounding box out of the image and return."""
    left = dict_obj['left']
    top = dict_obj['top']
    right = dict_obj['right']
    bottom = dict_obj['bottom']

    im = im[top:bottom, left:right, :]
    return im


def resize_bb(bounding_box, image_size):
    """Resize bounding box to fixed size."""
    img = np.asarray(bounding_box)
    if isinstance(image_size, tuple):
        img = cv2.resize(img, dsize=image_size)
    else:
        img = cv2.resize(img, dsize=(image_size, image_size))

    return np.array(img, dtype=np.uint8)


def calc_distance(v1, v2):
    """Calculate the Euclidean distance between two feature vectors."""
    return np.linalg.norm(v1 - v2)


def calc_cosine_sim(v1, v2):
    """Calculate the cosine similarity between two vectors."""
    v1_T = np.transpose(v1)
    v2_T = np.transpose(v2)
    return abs(np.dot(v1, v2_T) / (np.sqrt(np.dot(v1, v1_T)) * np.sqrt(np.dot(v2, v2_T))))


def show_frame_with_ids(frame, bboxes, ids, frame_num, seq_name, visual_location=None):
    """Visualize the video frame with bounding boxes and ids.

  Args:
    frame: Current video frame.
    bboxes: Bounding box data for objects in the current frame.
    ids: Identification number for the objects in the current frame.
  """
    # Convert frame and create the figure.
    figure_size = 8
    frame = np.asarray(frame, dtype=np.uint8)
    fig, ax = plt.subplots(figsize=(figure_size, int(figure_size/2)))

    # Remove the axis and add the image.
    ax.axis('off')
    ax.imshow(frame)

    # Add the bounding box and id to the frame.
    for i, bbox in enumerate(bboxes):
        left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]
        bbox_ = patches.Rectangle((left, top), right-left, bottom-top,
                                  linewidth=1,
                                  edgecolor='m',
                                  facecolor='none')
        ax.add_patch(bbox_)
        ax.text(left, top, ids[i],
                color='m',
                ha='right',
                va='bottom')

    # Show the frame with the bounding boxes and ids.
    if visual_location == None:
        plt.show(block=False)
        plt.pause(0.1)
        plt.close()
    else:
        visual_path = visual_location + '/'+ seq_name
        Path(visual_path).mkdir(parents=True, exist_ok=True)
        fig.savefig('{}/{}/frame{}.jpg'.format(visual_location, seq_name, frame_num))
        plt.close()


def show_frame_with_labels(frame, bboxes, labels, probs):
    """Visualize the video frame with bounding boxes, labels and probabilities.

  Args:
    frame: Current video frame.
    bboxes: Bounding box data for objects in the current frame.
    labels: Labels for the bounding boxes.
    probs: Probabilities for the bounding boxes.
  """
    text_labels = ["Cyclist", "Misc", "Person_sitting", "Tram", "Truck", "Van", "Car", "Person"]
    visualize_boxes(frame, bboxes, labels, probs, text_labels)

    # Show the frame with the bounding boxes, labels and probabilities.
    plt.imshow(frame.astype(np.uint8))
    plt.axis('off')

    plt.show(block=False)
    plt.pause(0.1)
    plt.close()


def check_acceptable_splits(dataset, train, val, test, allow_overfit):
    """Check if the chosen train/val/test splits are valid.

  Args:
    dataset: Type of dataset that is used.
    train: List of sequence numbers for training.
    val: List of sequence numbers for validation.
    test: List of sequence numbers for testing.
  """
    sequences = [*train, *val, *test]

    if not(allow_overfit):
        # Check if the splits overlap.
        if len(sequences) != len(set(sequences)):
            raise ValueError('Overlap between splits detected.')

    # Check if all chosen sequences are valid.
    if dataset == 'kitti':
        if np.any(np.array(sequences) < 0):
            raise ValueError('Sequence lower than 0 not possible.')
        elif np.any(np.array(sequences) > 20):
            raise ValueError('Sequence higher than 20 not possible.')
    else:
        raise ValueError('Function not defined for this dataset. The available dataset is: kitti')


def export_parameters(save_directory, learning_rate, l2_reg, l2_norm,
                      memory_length, memory_update, max_distance,
                      window_size, num_combi_per_obj_per_epoch,
                      sequences_train, sequences_val, sequences_test):
    with open(save_directory+'/config.txt', mode='w') as file:
        file.write('learning rate: '+str(learning_rate))
        file.write('\n')
        file.write('l2 reg: '+str(l2_reg))
        file.write('\n')
        file.write('l2 norm: '+str(l2_norm))
        file.write('\n')
        file.write('memory length: '+str(memory_length))
        file.write('\n')
        file.write('memory update: '+str(memory_update))
        file.write('\n')
        file.write('max distance: '+str(max_distance))
        file.write('\n')
        file.write('window size: '+str(window_size))
        file.write('\n')
        file.write('combi per obj per epoch: '+str(num_combi_per_obj_per_epoch))
        file.write('\n')
        file.write('seq train: '+str(sequences_train))
        file.write('\n')
        file.write('seq val: '+str(sequences_val))
        file.write('\n')
        file.write('seq test: '+str(sequences_test))
