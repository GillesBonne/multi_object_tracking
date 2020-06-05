import json
import logging
import os
import pickle

import h5py
import imageio
import numpy as np
import pandas as pd

logging.basicConfig(format='%(message)s', level=logging.INFO)

# List import files downloaded from http://www.cvlibs.net/datasets/kitti/eval_tracking.php.
scene_movie_dir = 'data_kitti/sequences'
scene_label_dir = 'data_kitti/annotations'
scene_images_dirs = sorted(np.array(os.listdir(scene_movie_dir)))
scene_labels = sorted(np.array(os.listdir(scene_label_dir)))

if not np.array_equal(scene_images_dirs, np.char.replace(scene_labels, '.txt', '')):
    raise ValueError('Number of scenes for images and labels are not identical.')

# Initialize label data structure.
labels_data = {}

# Export only 1 sequence?
export_only_first_sequence = False

# Naming of export files.
dataset_name = 'kitti'
filename_export_images = '_'.join([dataset_name, 'images.h5'])
filename_export_labels = '_'.join([dataset_name, 'labels.bin'])

# Get labels and write uncompressed images to h5 file.
with h5py.File(filename_export_images, 'w') as hf_images:
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

        # Read the first frame of the sequence to find the shape.
        im = imageio.imread(os.path.join(images_dir, images[0]))
        height, width, _ = im.shape

        # Create h5 file data structure for exporting the images.
        dset = hf_images.create_dataset(seq_name, (len(images), height, width, 3))
        ims = []
        first = 0
        last = 0
        for frame_num, image in enumerate(images):

            # Read images and append to list waiting for next export batch.
            im = imageio.imread(os.path.join(images_dir, image))
            ims.append(im)

            # Write to images file.
            batch_length = 200
            if len(ims) == batch_length:
                last += batch_length
                dset[first:last, :, :, :] = ims
                first += batch_length
                ims = []

            # Initialize frame specific label data structure.
            frame_labels_data = {}

            df_frame_labels = df_labels[df_labels['frame'] == frame_num]

            for index, row in df_frame_labels.iterrows():
                # Initialize object specific label data structure.
                object_labels_data = {}

                # Track in which frames each object is in.
                frames_per_id_dict[int(row['track_id'])].append(frame_num)

                # Save all the necessary labels.
                object_labels_data['track_id'] = int(row['track_id'])
                object_labels_data['occluded'] = int(row['occluded'])
                object_labels_data['type'] = row['type']
                object_labels_data['left'] = round(row['left'])
                object_labels_data['top'] = round(row['top'])
                object_labels_data['right'] = round(row['right'])
                object_labels_data['bottom'] = round(row['bottom'])

                # Save labels for the current object.
                object_name = 'obj'+str(int(row['track_id']))
                frame_labels_data[object_name] = object_labels_data

            # Save labels for the current frame.
            frame_name = 'frame'+str(frame_num)
            seq_labels_data[frame_name] = frame_labels_data

            # Log the progress.
            if frame_num % 100 == 0:
                logging.info(frame_num)

        # Write images to .h5 file.
        dset[first:, :, :, :] = ims

        # Save in which frame a certain object is in.
        seq_labels_data['frames_per_id'] = frames_per_id_dict

        # Save labels for the current sequence.
        labels_data[seq_name] = seq_labels_data

        # Export only 1 sequence.
        if export_only_first_sequence:
            break

# Export labels to Python pickle file.
with open(filename_export_labels, 'wb') as file:
    pickle.dump(labels_data, file)
