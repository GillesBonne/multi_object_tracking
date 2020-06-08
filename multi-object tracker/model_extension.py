"""Multi-object tracker as extension for the single object tracker."""

import numpy as np

from yolo.config import ConfigParser
from utils import resize_bb

class MultiTrackNet():
    """Extension for the TrackNet identification network. Creating an end-to-end
    tracking network with pretrained YOLO-v3 as detection network and the provided 
    TrackNet as re-identification network.

    YOLO-v3 website: https://github.com/penny4860/tf2-eager-yolo3 
    """
    def __init__(self, trained_tracknet_network, 
                            config_path='pretrained_model/kitti.json', bbox_size=128):
        # Get the TrackNet model
        self.identifier = trained_tracknet_network
        self.config_path = config_path
        self.bbox_size = bbox_size

        # Create the YOLO-v3 model and load the weights
        config_parser = ConfigParser(self.config_path)
        model = config_parser.create_model(skip_detect_layer=False)
        self.detector = config_parser.create_detector(model)

    def __call__(self, image, prob_threshold=0.5):
        # Check if the input is only single image
        image_dims = image.shape
        assert len(image_dims) == 3, 'Error: more than one image as input!'

        # Run the image through the detector network
        boxes, labels, probs = self.detector.detect(image.astype(int), prob_threshold)

        # Loop over the bouding boxes
        bbox_batch = []
        for box in boxes:
            box = [int(max(min(x, max(image_dims)), 0)) for x in box]
            bbox = image[box[1]:box[3], box[0]:box[2], :]
            bbox_batch.append(resize_bb(bbox, self.bbox_size))

        # Run bounding boxes through re-identification network
        if len(bbox_batch) != 0:
            embeddings = self.identifier(np.array(bbox_batch), training=False)
        else:
            embeddings = []

        # Return vector with embeddings (and more)
        return embeddings, boxes, labels, probs

