import pickle

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # Import Python pickle file.
    with open('kitti_labels.bin', 'rb') as file:
        labels = pickle.load(file)

    amount_of_invisible_frames = []
    for sequence in labels:
        frames_per_id = labels[sequence]['frames_per_id']
        for id in frames_per_id:
            frames = frames_per_id[id]
            for i in range(len(frames)):
                if i == 0:
                    continue
                else:
                    if frames[i] != frames[i-1] + 1:
                        amount_of_invisible_frames.append(frames[i]-frames[i-1]-1)

    print('Length of temporary complete occlusion of an object.')
    print(amount_of_invisible_frames)

    for seq in labels:
        seq_labels = labels[seq]
        ids = list(seq_labels['frames_per_id'].keys())

        occlusion_data = np.ones((len(seq_labels)-1, len(ids)))*-1

        for j, id in enumerate(ids):
            substituted_3 = False
            for i, frame in enumerate(seq_labels):
                frame_labels = seq_labels[frame]
                if frame == 'frames_per_id':
                    if i != len(seq_labels)-1:
                        raise ValueError('Order of data has changed. Frames per id should be last.')
                    break
                else:
                    for obj in frame_labels:
                        obj_labels = frame_labels[obj]
                        if obj_labels['track_id'] == id:
                            occlusion_data[i, j] = obj_labels['occluded']

                    # Set unknown occluded value to 2.
                    if occlusion_data[i, j] == 3:
                        occlusion_data[i, j] = 2

                    # If object is not found then it is fully occluded.
                    if occlusion_data[i, j] == -1:
                        occlusion_data[i, j] = 3

        plt.imshow(occlusion_data)
        plt.title('Sequence: {}'.format(seq))
        plt.colorbar()
        plt.show()
