import logging
import os

import numpy as np
import pandas as pd

logging.basicConfig(format='%(message)s', level=logging.INFO)

scene_movie_dir = 'data_kitti/sequences'
scene_label_dir = 'data_kitti/annotations'
scene_images_dirs = sorted(np.array(os.listdir(scene_movie_dir)))
scene_labels = sorted(np.array(os.listdir(scene_label_dir)))

if not np.array_equal(scene_images_dirs, np.char.replace(scene_labels, '.txt', '')):
    raise ValueError('Number of scenes for images and labels are not identical.')

# Initialize label data structure.
labels_data = {}

total_num_objects = 0

# Get labels and write uncompressed images to h5 file.
for scene_index, (scene_images_dir, scene_label) in enumerate(zip(scene_images_dirs, scene_labels)):
    logging.info(f'Sequence: {scene_index}')

    # Initialize sequence specific label data structure.
    seq_labels_data = {}
    seq_num = int(scene_images_dir)
    seq_name = 'seq'+str(seq_num)

    # List all the image directories and label files.
    images_dir = os.path.join(scene_movie_dir, scene_images_dir)
    images = sorted(np.array(os.listdir(images_dir)))
    label_file = os.path.join(scene_label_dir, scene_label)
    if not np.all(np.char.endswith(images, '.png')):
        raise ValueError('Non png file encountered in scene.')

    # Read labels from text file.
    header_names = ['frame', 'track_id', 'type', 'truncated', 'occluded', 'alpha',
                    'left', 'top', 'right', 'bottom',
                    'dim_heigth', 'dim_width', 'dim_length',
                    'loc_x', 'loc_y', 'loc_z',
                    'rotation_y']
    df_labels = pd.read_csv(label_file, sep=' ', names=header_names)
    df_labels = df_labels[df_labels['type'] != 'DontCare'].reset_index(drop=True)
    df_labels = df_labels.drop(columns=['truncated', 'alpha',
                                        'dim_heigth', 'dim_width', 'dim_length',
                                        'loc_x', 'loc_y', 'loc_z', 'rotation_y'])

    # Get track_ids.
    track_ids = sorted(np.array(list(set(df_labels['track_id']))))
    frames_per_id_dict = {}
    [frames_per_id_dict.setdefault(id, []) for id in track_ids]

    total_num_objects += len(track_ids)
    print(len(track_ids))
print('Total number of objects: {}'.format(total_num_objects))
