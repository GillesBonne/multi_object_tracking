import pickle

import numpy as np

# Import Python pickle file.
with open('kitti_first_seq_labels.bin', 'rb') as file:
    labels_dict = pickle.load(file)

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

        # Choose random initial frame.
        frame_anchor, frame_positive = np.random.choice(occurrences, size=2, replace=False)

        # Random choice from random choice of object_ids not equal to object_id.
        object_id_compare = np.random.choice(object_ids_not_current)
        occurrences_compare = labels_dict[seq_key]['frames_per_id'][object_id_compare]
        frame_negative = np.random.choice(occurrences_compare)

        # Print choices.
        print('Object {}, anchor frame {}, positive frame {}.'.format(
            object_id, frame_anchor, frame_positive))
        print('Negative object {}, negative frame {}'.format(
            object_id_compare, frame_negative))
