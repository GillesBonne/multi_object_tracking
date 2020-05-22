"""Evaluation functions for the multi object tracker."""

from __future__ import absolute_import, division, print_function

import numpy as np

import motmetrics as mm


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

    def update(self, obj_ids, hyp_ids, obj_bbox, hyp_bbox, frameid=None):
        """ Update the accumulator with frame objects and hypotheses.

      Args:
        obj_ids: List with object ids with format [id1, id2, ..., idn]
        hyp_ids: List with hypothesis ids with format [id1, id2, ..., idn]
        obj_bbox: Numpy array with object bounding boxes (x1, y1, x2, y2) in rows.
        hyp_bbox: Numpy array with hypothesis bounding boxes (x1, y1, x2, y2) in rows.
        frameid: Frame id when auto_id is set to False.
      """
        # Convert from (x1, y1, x2, y2) to (x, y, w, h) format
        for i, row in enumerate(obj_bbox):
            width = obj_bbox[i, 2] - obj_bbox[i, 0]
            height = obj_bbox[i, 3] - obj_bbox[i, 1]

            assert width >= 0, 'Error in bounding box values!'
            assert height >= 0, 'Error in bounding box values!'
            obj_bbox[i, 2:4] = [width, height]

        for i, row in enumerate(hyp_bbox):
            width = hyp_bbox[i, 2] - hyp_bbox[i, 0]
            height = hyp_bbox[i, 3] - hyp_bbox[i, 1]

            assert width >= 0, 'Error in bounding box values!'
            assert height >= 0, 'Error in bounding box values!'
            hyp_bbox[i, 2:4] = [width, height]

        # Calculate the distance matrix
        distances = mm.distances.iou_matrix(obj_bbox, hyp_bbox, max_iou=1.)

        # Update the accumulator
        self.accumulator.update(obj_ids, hyp_ids, distances, frameid=frameid)

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
