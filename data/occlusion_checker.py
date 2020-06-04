import copy
import os

import numpy as np
import xmltodict
from tqdm import tqdm

path = 'data_detrac/annotations/'
annotations = os.listdir(path)

for annotation in tqdm(annotations):
    with open(path+annotation, 'r') as file:
        annotation_data = xmltodict.parse(file.read())

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
                all_ids.add(int(object['@id']))
    track_ids = sorted(np.array(list(all_ids)))
    frames_per_id_dict = {}
    [frames_per_id_dict.setdefault(id, []) for id in track_ids]

    for i, frame in enumerate(annotation_data['sequence']['frame']):
        frame_num = int(frame['@num'])
        num_objects = int(frame['@density'])
        if num_objects == 0:
            raise ValueError('No objects in frame.')
        elif num_objects == 1:
            object = frame['target_list']['target']
            id = int(object['@id'])
            # Track in which frames each object is in.
            frames_per_id_dict[id].append(frame_num)
        else:
            for object in frame['target_list']['target']:
                id = int(object['@id'])

                # Track in which frames each object is in.
                frames_per_id_dict[id].append(frame_num)
    for key in frames_per_id_dict:
        obj_appearances = frames_per_id_dict[key]
        for i, obj_appearance in enumerate(obj_appearances):
            if i == 0:
                previous_appearance = copy.deepcopy(obj_appearance)
                continue
            else:
                if obj_appearance != previous_appearance + 1:
                    print('yes')
                    print('yes')
                    print(obj_appearance)
                    print('yes')
                    print(obj_appearances)
                    exit()
                previous_appearance = copy.deepcopy(obj_appearance)
