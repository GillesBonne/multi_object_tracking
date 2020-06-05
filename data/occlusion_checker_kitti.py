import pickle

if __name__ == '__main__':
    # Import Python pickle file.
    with open('kitti_labels.bin', 'rb') as file:
        labels = pickle.load(file)

    for sequence in labels:
        frames_per_id = labels[sequence]['frames_per_id']
        for id in frames_per_id:
            frames = frames_per_id[id]
            for i in range(len(frames)):
                if i == 0:
                    continue
                else:
                    if frames[i] != frames[i-1] + 1:
                        print('Object is completely occluded')
                        print('Sequence: ', sequence)
                        print('ID: ', id)
                        print('Frame: ', frames[i])
                        print(frames)
                        print()
