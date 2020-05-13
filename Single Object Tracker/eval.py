"""Evaluation functions for the multi object tracker."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import motmetrics as mm
import numpy as np


class MOTMetric():
    """Wrapper around MOT metric implementation from GitHub.
      
  Multi Object Tracking metric as described by:
    Evaluating Multiple Object Tracking Performance: The CLEAR MOT Metrics
    https://dl.acm.org/doi/pdf/10.1155/2008/246309
    by Keni Bernardin and Rainer Stiefelhagen, Apr 2008

  GitHub website: https://github.com/cheind/py-motmetrics
  """
    def __init__(self, auto_id=False):
        """ Create an accumulator that will be updated during each frame."""
        self.accumulator = mm.MOTAccumulator(auto_id=auto_id)
        self.metric_host = mm.metrics.create()

    def update(self, object_ids, hypothesis_ids, objects, hypotheses, frameid=None):
        """ Update the accumulator with frame objects and hypotheses.
        
      Args:
        object_ids: List with object ids with format [id1, id2, ..., idn]
        hypothesis_ids: List with hypothesis ids with format [id1, id2, ..., idn]
        objects: Numpy array with object bounding boxes (x1, y1, x2, y2) in rows.
        hypotheses: Numpy array with hypothesis bounding boxes (x1, y1, x2, y2) in rows.
        frameid: Frame id when auto_id is set to False.
      """
        # Convert from (x1, y1, x2, y2) to (x, y, w, h) format
        for i, row in enumerate(objects):
            width = objects[i,2] - objects[i,0]
            height = objects[i,3] - objects[i,1]
            
            assert width >= 0, 'Error in bounding box values!'
            assert height >= 0, 'Error in bounding box values!'
            objects[i,2:4] = [width, height]

        for i, row in enumerate(hypotheses):
            width = hypotheses[i,2] - hypotheses[i,0]
            height = hypotheses[i,3] - hypotheses[i,1]
            
            assert width >= 0, 'Error in bounding box values!'
            assert height >= 0, 'Error in bounding box values!'
            hypotheses[i,2:4] = [width, height]

        # Calculate the distance matrix
        distances = mm.distances.iou_matrix(objects, hypotheses, max_iou=1.)

        # Update the accumulator
        self.accumulator.update(object_ids, hypothesis_ids, distances, frameid=frameid)

    def get_MOTA(self):
        """Return the Multi Object Tracking Accuracy."""
        summary = self.metric_host.compute(self.accumulator, 
            metrics='mota', return_dataframe=False)
        return summary['mota']

    def get_MOTP(self):
        """Return the Multi Object Tracking Precision."""
        summary = self.metric_host.compute(self.accumulator, 
            metrics='motp', return_dataframe=False)
        return summary['motp']

    def get_metric(self, metric_list):
        """Return the Multi Object Tracking metrics defined in 'metric_list'."""
        summary = self.metric_host.compute(self.accumulator, 
            metrics=metric_list, return_dataframe=False)
        return summary

