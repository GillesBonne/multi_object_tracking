import pickle

# Import Python pickle file.
with open('kitti_first_seq_labels.bin', 'rb') as file:
    labels = pickle.load(file)
