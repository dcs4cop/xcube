# The MIT License (MIT)
# Copyright (c) 2021 by the xcube development team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numba as nb
import numpy as np


@nb.jit(nopython=True, nogil=True, parallel=True, cache=True)
def compute_ij_boxes(x_image: np.ndarray,
                     y_image: np.ndarray,
                     xy_boxes: np.ndarray,
                     xy_border: float,
                     ij_border: int,
                     ij_boxes: np.ndarray):
    h = x_image.shape[0]
    w = x_image.shape[1]
    n = xy_boxes.shape[0]
    for k in nb.prange(n):
        ij_bbox = ij_boxes[k]
        xy_bbox = xy_boxes[k]
        x_min = xy_bbox[0] - xy_border
        y_min = xy_bbox[1] - xy_border
        x_max = xy_bbox[2] + xy_border
        y_max = xy_bbox[3] + xy_border
        for j in range(h):
            for i in range(w):
                x = x_image[j, i]
                if x_min <= x <= x_max:
                    y = y_image[j, i]
                    if y_min <= y <= y_max:
                        i_min = ij_bbox[0]
                        j_min = ij_bbox[1]
                        i_max = ij_bbox[2]
                        j_max = ij_bbox[3]
                        ij_bbox[0] = i if i_min < 0 else min(i_min, i)
                        ij_bbox[1] = j if j_min < 0 else min(j_min, j)
                        ij_bbox[2] = i if i_max < 0 else max(i_max, i)
                        ij_bbox[3] = j if j_max < 0 else max(j_max, j)
        if ij_border != 0 and ij_bbox[0] != -1:
            i_min = ij_bbox[0] - ij_border
            j_min = ij_bbox[1] - ij_border
            i_max = ij_bbox[2] + ij_border
            j_max = ij_bbox[3] + ij_border
            if i_min < 0:
                i_min = 0
            if j_min < 0:
                j_min = 0
            if i_max >= w:
                i_max = w - 1
            if j_max >= h:
                j_max = h - 1
            ij_bbox[0] = i_min
            ij_bbox[1] = j_min
            ij_bbox[2] = i_max
            ij_bbox[3] = j_max
