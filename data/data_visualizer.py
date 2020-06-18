import pickle

import h5py
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

labels_file = 'kitti_labels.bin'
images_file = 'kitti_images.h5'

# Import Python pickle file.
with open(labels_file, 'rb') as file:
    labels_dict = pickle.load(file)

seq = 12

# Get the frame data.
with h5py.File(images_file, 'r') as data:
    for i, frame in enumerate(data['seq'+str(seq)]):
        gt_labels = labels_dict['seq'+str(seq)]['frame'+str(i)]
        obj_ids, obj_bbs = [], []
        for label in gt_labels.values():
            obj_ids.append(label['track_id'])
            obj_bbs.append([label['left'], label['top'],
                            label['right'], label['bottom']])

        frame_show = np.asarray(frame, dtype=np.uint8)
        # Convert frame and create the figure.
        figure_size = 14
        fig, ax = plt.subplots(figsize=(figure_size, int(figure_size/2)))

        # Remove the axis and add the image.
        ax.axis('off')
        ax.imshow(frame_show)

        # Add the bounding box and id to the frame.
        for i, bbox in enumerate(obj_bbs):
            left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]
            bbox_ = patches.Rectangle((left, top), right-left, bottom-top,
                                      linewidth=1,
                                      edgecolor='m',
                                      facecolor='none')
            ax.add_patch(bbox_)
            ax.text(left, top, obj_ids[i],
                    color='m',
                    ha='right',
                    va='bottom')
        plt.show()
