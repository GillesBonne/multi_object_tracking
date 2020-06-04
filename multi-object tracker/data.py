"""Data handling functions for the video sequences."""

from __future__ import absolute_import, division, print_function

import itertools
import pickle

import numpy as np


def get_combinations(labels_file):
    """Get the combinations for anchor, positive and negative sample."""

    # Import Python pickle file.
    with open(labels_file, 'rb') as file:
        labels_dict = pickle.load(file)

    # Create combinations for every epoch.
    combinations = []
    for seq_key in labels_dict:

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

            # Check if object is not seen in all sequential frames.
            for i, occurrence in enumerate(occurrences):
                if i == 0:
                    continue
                else:
                    if occurrences[i] != occurrences[i-1] + 1:
                        print('There is a case for which an object is not seen in all sequential frames.')
                        print('The object occurs in the following frames:')
                        print(occurrences)

            # Specify the window size of the object scan across the video.
            window_size = 5
            if window_size < 2:
                raise ValueError('Window size should be larger than 2.')

            # Get all window options with the specified window size.
            frame_combinations = set()
            for i in range(len(occurrences)):
                frame_window = occurrences[i:i+window_size]
                if len(frame_window) < window_size:
                    continue
                else:
                    combs = set(list(itertools.combinations(frame_window, 2)))
                    frame_combinations.update(combs)

            num_combinations_per_seq_per_obj = 10
            frame_combinations = np.array(list(frame_combinations))
            frame_indices = np.random.choice(
                len(frame_combinations), num_combinations_per_seq_per_obj, replace=True)
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
        len(get_combinations('../data/detrac_first_seq_labels.bin'))))
