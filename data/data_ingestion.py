import pickle

with open('kitti_first_seq_labels.bin', 'rb') as file:
    your_dict = pickle.load(file)
    print(your_dict['seq0']['frames_per_id'][0])
