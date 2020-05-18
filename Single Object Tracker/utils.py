"""Utilary functions for the single object tracker."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from PIL import Image


def rescale_bb(bounding_box, size):
    """Rescale boudning box to fixed size."""
    img = Image.fromarray(bouding_box)
    new_img = img.resize(size) if isinstance(size, tuple) else img.resize((size, size))
    return np.array(new_img)


def calc_distance(vector1, vector2):
    """Calculate the distance between two feature vectors."""
    return np.linalg.norm(vector1 - vector2)


def re_identification(embeds_dict, new_embeds, max_dist=1):
    """Link the current embeddings to the embedding of the previous frame.

  Args:
    embeds_dict: Embeddings of all known bouding boxes as key and there 
      identification as value. For inital frame, the 'embeds_dict' is empty.
    new_embeds: New embeddings that have to be linked and assigned with id.
    max_dist: Maximum distance between embeddings before they are not linked. 
  """
    # Put the embedding in list two preserve there order
    embeds = list(embeds_dict.keys())

    # Calculate the distance between every combination
    dist_matrix = np.empty([len(new_embeds), len(embeds)])

    print(len(embeds))  # REMOVE
    
    for i, new_embed in enumerate(new_embeds):
        for j, embed in enumerate(embeds):
            # Put penalties for the distance calculation here
            dist = calc_distance(new_embed, embed)
            dist_matrix[i,j] = dist

    # Link every new embedding to embedding with the shortest distance
    new_embeds_dict = {}
    for i, row in enumerate(dist_matrix):
        shortest_dist = np.min(row)

        print(shortest_dist)  # REMOVE

        if shortest_dist > max_dist:
            # Distance too large, assign new id
            new_id = max(embeds_dict.values())+1
            new_embeds_dict[new_embeds[i]] = new_id
            embeds_dict[new_embeds[i]] = new_id
            continue

        # Find the best embedding and the identification
        embed_ = embeds[np.where(row==shortest_dist)]
        id_ = embeds_dict[embed_match]
        new_embeds_dict[new_embeds[i]] = id_

    return embeds_dict, new_embeds_dict


embed1 = np.random.rand(1,128).tolist()
embed2 = np.random.rand(1,128).tolist()
embed3 = np.random.rand(1,128).tolist()
embed4 = np.random.rand(1,128).tolist()
embed5 = np.random.rand(1,128).tolist()

embeds_dict = {embed1:1, embed2:2, embed3:3, embed1:4}
new_embeds = [embed3, embed4, embed5]

embeds_dict, new_embeds_dict = re_identification(embeds_dict, new_embeds, max_dist=1)

print(embeds_dict)
print(new_embeds_dict)



    # Make embeddings for every bouding box in the initial frame and assign id

    # Make embedding for every bounding box in the next frame

    # Link the bounding boxes by using the distance between embeddings
    # and assign the corresponding id

    # If distance is too large, give new id and add to list/dict






    # # Calculate the distance between every combination
    # dist_matrix = np.empty([len(prev_embeds), len(curr_embeds)])
    
    # for i, prev_embed in enumerate(prev_embeds):
    #     for j, curr_embed in enumerate(curr_embeds):
    #         # Put the linking algorithm here
    #         dist = calc_distance(prev_embed, curr_embed)
    #         dist_matrix[i,j] = dist

    # # Determine the pairs with the shortest distance
    # min_dist = np.min(a[np.nonzero(a)])