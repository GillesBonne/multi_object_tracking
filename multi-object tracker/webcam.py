"""Run the tracking network on your webcam data."""

import cv2
import numpy as np

from model import TrackNet
from model_v2 import TrackNetV2
from model_extension import MultiTrackNet
from embeds import EmbeddingsDatabase


# Load the network
save_directory = 'good_model'
model_path = save_directory + '/saved_model.ckpt'

model = TrackNet(use_bias=False, l2_reg=0.001, use_dropout=True)
model.load_weights(model_path)

# Attach the detector
tracker = MultiTrackNet(model)

# Create the embeddings database
embeds_database = EmbeddingsDatabase(memory_length=30, memory_update=0.75)

# Connect with the webcam
cv2.namedWindow("Webcam")
vc = cv2.VideoCapture(0)

# Try to get the first frame
if vc.isOpened():
    rval, frame = vc.read()
else:
    rval = False

# Run the tracking network on the webcam feed
while rval:
    cv2.imshow("Webcam", frame)
    rval, frame = vc.read()

    embeddings, boxes, labels, probs = tracker(frame)
    boxes = np.array(boxes, dtype=int)
    id_numbers = embeds_database.match_embeddings(embeddings, max_distance=0.5)

    if id_numbers:
        print(id_numbers)
    
    # Stop the loop by pressing the ESC key
    key = cv2.waitKey(20)
    if key == 27:
        break
    else:
        for i, box in enumerate(boxes):
            left, top, right, bottom = box[0], box[1], box[2], box[3]
            cv2.rectangle(frame, (left, top), (right, bottom), 
                color=(0, 255, 0), thickness=2)
            # cv2.putText(frame, id_numbers[i], (left, top), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
            #     fontScale=1, color=(0, 255, 0), thickness=2)

# Release the webcam
vc.release()
cv2.destroyWindow("Webcam")

