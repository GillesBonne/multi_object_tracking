"""Utilary functions for the single object tracker."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from collections import OrderedDict


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


def re_identification(embeds_dict, new_embeds, method='Euclidean', max_dist=0.2, update=False):
    """Link the current embeddings to the embedding of the previous frame.

  Args:
    embeds_dict: Dict with the ids as keys and the embeddings as values.
      For the initial frame, the 'embeds_dict' object is an empty dict. 
    new_embeds: New embeddings that have to be linked and assigned with id.
    method: Method to use for comparison: 'Euclidean' or 'cosine'.
    max_dist: Maximum distance between embeddings before they are not linked.
    update: When set to True, the 'embeds_dict' is updated with the new embeddings.

    NOTE: Two embeddings can get the same identification!  
  """
    # Convert dict to ordered dict to preserve the order
    embeds_dict = OrderedDict(embeds_dict)
    embeds_list = embeds_dict.values()

    # Select the similarity function
    if method == 'Euclidean':
        function = calc_distance
    elif method == 'cosine':
        function = calc_cosine_sim

    # Find the best matching embedding
    ids_list = []
    for new_embed in new_embeds:
        if not embeds_list:
            # No embedding in 'embeds_dict' to compare with
            new_id = max(embeds_dict.keys())+1 if embeds_dict else 0
            embeds_dict[new_id] = new_embed
            ids_list.append(new_id)
            continue

        # Calculate the similarity between embeddings
        similarity = []
        for embed in embeds_list:
            similarity.append(function(new_embed, embed))

        min_value = min(similarity)
        index = similarity.index(min_value)

        if min_value > max_dist:
            # Distance too large, assign new id
            new_id = max(embeds_dict.keys())+1
            embeds_dict[new_id] = new_embed
            ids_list.append(new_id)
            continue

        id_match = list(embeds_dict.keys())[index]
        ids_list.append(id_match)

        if update:
            # Update the dict with the new embedding
            embeds_dict[id_match] = new_embed

    return embeds_dict, ids_list


def show_frame_with_bb(frame, bboxes, ids, fps=30):
    """Visualize the video frame with bounding boxes and ids.

  Args:
    frame: Current video frame. 
    bboxes: Bounding box data for objects in the current frame.
    ids: Identification number for the objects in the current frame.
    fps: Frame per second, used to determine the pause in between frames. 
  """
    # Convert frame and create the figure
    figure_size = 8
    frame = np.asarray(frame, dtype=np.uint8)
    fig, ax = plt.subplots(figsize=(figure_size, int(figure_size/2)))

    # Remove the axis and add the image
    ax.axis('off')
    ax.imshow(frame)

    # Add the bounding box and id to the frame
    for i, bbox in enumerate(bboxes):
        left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]
        bbox_ = patches.Rectangle((left, top), right-left, bottom-top, 
            linewidth=2, 
            edgecolor='m', 
            facecolor='none')
        ax.add_patch(bbox_)
        ax.text(left, top, ids[i],
            color='m', 
            ha='right', 
            va='bottom')

    # Show the frame with the bounding boxes and ids
    plt.show(block=False)
    plt.pause(1/fps)
    plt.close()


