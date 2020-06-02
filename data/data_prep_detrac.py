import logging
import os
import pickle
from pprint import pprint

import h5py
import imageio
import matplotlib.pyplot as plt
import numpy as np
import xmltodict

logging.basicConfig(format='%(message)s', level=logging.INFO)

# Download data and combine into annotations and sequences folders.
# http://detrac-db.rit.albany.edu/Data/DETRAC-train-data.zip
# http://detrac-db.rit.albany.edu/Data/DETRAC-Test-Annotations-XML.zip
# http://detrac-db.rit.albany.edu/Data/DETRAC-test-data.zip
# http://detrac-db.rit.albany.edu/Data/DETRAC-Train-Annotations-XML.zip


def get_object_info(object, frame_labels_data, frames_per_id_dict, visualize):
    object_labels_data = {}

    id = int(object['@id'])
    left = round(float(object['box']['@left']))
    top = round(float(object['box']['@top']))
    width = round(float(object['box']['@width']))
    height = round(float(object['box']['@height']))
    type = str(object['attribute']['@vehicle_type'])

    right = left + width
    bottom = top + height

    # Track in which frames each object is in.
    frames_per_id_dict[id].append(frame_num)

    # Save all the necessary labels.
    object_labels_data['track_id'] = id
    object_labels_data['type'] = type
    object_labels_data['left'] = left
    object_labels_data['top'] = top
    object_labels_data['right'] = right
    object_labels_data['bottom'] = bottom

    # Save labels for the current object.
    object_name = 'obj'+str(id)
    frame_labels_data[object_name] = object_labels_data

    if visualize:
        xy = (left, top)
        rect = plt.Rectangle(xy=xy, width=width, height=height,
                             fill=False, edgecolor='red')
        ax.add_patch(rect)

        # object_text = 'Type: {}\nID: {}'.format(type, id)
        object_text = 'ID: {}'.format(id)
        ax.text(*xy, object_text, fontsize=9, color='red')

    return frame_labels_data, frames_per_id_dict


visualize = True

path_to_annotations = 'data_detrac/annotations'
path_to_sequences = 'data_detrac/sequences'

annotations = sorted(os.listdir(path_to_annotations))
sequences = sorted(os.listdir(path_to_sequences))

if not np.array_equal(np.array(sequences), np.char.replace(np.array(annotations), '.xml', '')):
    raise ValueError('Sequences of labels and images are not identical.')
else:
    logging.info('Identical sequences check passed.')

# Initialize label data structure.
labels_data = {}

# Export only 1 sequence?
export_only_first_sequence = False

# Maximum filesize.
FILE_MAX_GB = 10

# Naming of export files.
dataset_name = 'detrac_first_seq'
filename_export_images = '_'.join([dataset_name, 'images.h5'])
filename_export_labels = '_'.join([dataset_name, 'labels.bin'])

# Get labels and write uncompressed images to h5 file.
with h5py.File(filename_export_images, 'w') as hf_images:
    for sequence in sequences:
        if sequence != 'MVI_39511':
            continue

        # Maximium file size.
        if os.path.getsize(filename_export_images)/10**9 > FILE_MAX_GB:
            break

        logging.info(f'Sequence: {sequence}.')

        annotation_path = ''.join([os.path.join(path_to_annotations, sequence), '.xml'])

        # Initialize sequence specific label data structure.
        seq_labels_data = {}

        with open(annotation_path, 'r') as annotation_file:
            annotation_data = xmltodict.parse(annotation_file.read())

            # Get track_ids.
            all_ids = set()
            for frame in annotation_data['sequence']['frame']:
                num_objects = int(frame['@density'])
                if num_objects == 1:
                    all_ids.add(int(frame['target_list']['target']['@id']))
                elif num_objects == 0:
                    print(sequence)
                    print(frame)
                    raise ValueError('No objects in frame.')
                else:
                    for object in frame['target_list']['target']:
                        print(frame['@num'])
                        print(object)
                        print(int(object['@id']))
                        all_ids.add(int(object['@id']))
                        print('appended')
            track_ids = sorted(np.array(list(all_ids)))
            frames_per_id_dict = {}
            [frames_per_id_dict.setdefault(id, []) for id in track_ids]

            # Read the first frame of the sequence to find the shape.
            im_name = 'img'+f"{int(annotation_data['sequence']['frame'][0]['@num']):05d}"+'.jpg'
            image_path = os.path.join(path_to_sequences, sequence, im_name)
            im = imageio.imread(image_path)
            im_height, im_width, _ = im.shape

            # Create h5 file data structure for exporting the images.
            images = os.listdir(os.path.join(path_to_sequences, sequence))
            dset = hf_images.create_dataset(sequence, (len(images), im_height, im_width, 3))
            ims = []
            first = 0
            last = 0

            for i, frame in enumerate(annotation_data['sequence']['frame']):
                frame_num = int(frame['@num'])-1
                im_name = 'img'+f'{(frame_num+1):05d}'+'.jpg'
                image_path = os.path.join(path_to_sequences, sequence, im_name)
                im = imageio.imread(image_path)
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

                if visualize:
                    fig, ax = plt.subplots()
                    ax.imshow(im)

                num_objects = int(frame['@density'])
                if num_objects == 1:
                    object = frame['target_list']['target']
                    frame_labels_data, frames_per_id_dict = get_object_info(
                        object, frame_labels_data, frames_per_id_dict, visualize)
                elif num_objects == 0:
                    raise ValueError('No objects in frame.')
                else:
                    for object in frame['target_list']['target']:
                        frame_labels_data, frames_per_id_dict = get_object_info(
                            object, frame_labels_data, frames_per_id_dict, visualize)

                if visualize:
                    plt.show()

                # Save labels for the current frame.
                frame_name = 'frame'+str(frame_num)
                seq_labels_data[frame_name] = frame_labels_data

                # Log the progress.
                if i % 100 == 0:
                    logging.info(frame_num)

        # Write images to .h5 file.
        dset[first:, :, :, :] = ims

        # Save in which frame a certain object is in.
        seq_labels_data['frames_per_id'] = frames_per_id_dict

        # Save labels for the current sequence.
        labels_data[sequence] = seq_labels_data

        if export_only_first_sequence:
            break

# Export labels to Python pickle file.
with open(filename_export_labels, 'wb') as file:
    pickle.dump(labels_data, file)
