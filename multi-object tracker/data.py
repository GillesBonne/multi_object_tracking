"""Data handling functions for the video sequences."""

from __future__ import absolute_import, division, print_function

import itertools
import pickle

import numpy as np


def get_combinations(labels_file, sequences, window_size, num_combi_per_obj_per_epoch):
    """Get the combinations for anchor, positive and negative sample."""

    # Import Python pickle file.
    with open(labels_file, 'rb') as file:
        labels_dict = pickle.load(file)

    # Create combinations for every epoch.
    combinations = []
    for seq_key in labels_dict:

        seq_num = int(''.join(ch for ch in seq_key if ch.isdigit()))
        if seq_num in sequences:

            # Get all the object ids.
            object_ids = list(labels_dict[seq_key]['frames_per_id'].keys())

            for object_id in object_ids:

                # Get object ids to be able to compare to.
                object_ids_not_current = object_ids.copy()
                object_ids_not_current.remove(object_id)

                # Get occurrences for this object id.
                occurrences = labels_dict[seq_key]['frames_per_id'][object_id]

                # Can not compare if there is nothing to compare to.
                if len(occurrences) < 2:
                    continue

                if window_size < 2:
                    raise ValueError('Window size should be larger than 1.')

                # Get all window options with the specified window size.
                frame_combinations = set()
                for i in range(len(occurrences)):
                    frame_window = occurrences[i:i+window_size]
                    if len(frame_window) < 2:
                        continue
                    else:
                        combs = set(list(itertools.combinations(frame_window, 2)))
                        frame_combinations.update(combs)

                num_combi_per_obj_per_epoch = 10
                frame_combinations = np.array(list(frame_combinations))
                frame_indices = np.random.choice(
                    len(frame_combinations), num_combi_per_obj_per_epoch, replace=True)
                frame_combinations = frame_combinations[frame_indices]

                # Get all the combination from the window options.
                for frame_first, frame_second in frame_combinations:
                    frame_anchor, frame_positive = np.random.choice(
                        [frame_first, frame_second], 2, replace=False)

                    object_id_compare = np.random.choice(object_ids_not_current)
                    occurrences_compare = labels_dict[seq_key]['frames_per_id'][object_id_compare]
                    frame_negative = np.random.choice(occurrences_compare)

                    combinations.append([seq_key, object_id, frame_anchor, frame_positive,
                                         object_id_compare, frame_negative])

    return combinations


if __name__ == '__main__':
    print('Number of combinations per epoch: {}.'.format(
        len(get_combinations('../data/kitti_labels.bin', [1, 8], window_size=35, num_combi_per_obj_per_epoch=10))))
