"""Utilary functions for the single object tracker."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from PIL import Image


def rescale_bb(bounding_box, size):
    """Rescale boudning box to fixed size."""
    img = Image.fromarray(bouding_box)
    new_img = img.resize(size) if isinstance(size, tuple) else img.resize((size, size))
    return numpy.array(new_img)


def calc_distance(vector1, vector2):
    """Calculate the distance between two feature vectors."""
    return np.linalg.norm(vector1 - vector2)