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
from typing import Tuple, Union

import dask.array as da
import numba as nb
import numpy as np
import xarray as xr


@nb.jit(nopython=True, nogil=True, parallel=True, cache=True)
def compute_ij_bboxes(x_image: np.ndarray,
                      y_image: np.ndarray,
                      xy_boxes: np.ndarray,
                      xy_border: float,
                      ij_border: int,
                      ij_boxes: np.ndarray):
    """
    Compute bounding boxes in the image's i,j coordinates from given
    x,y coordinates *x_image*, *y_image* and bounding boxes
    in x,y coordinates *xy_boxes*.

    *ij_boxes* must be pre-allocated to match shape of *xy_boxes* and initialised
    with negative integers.

    :param x_image: The x coordinates image. A 2D array of shape (height, width).
    :param y_image: The y coordinates image. A 2D array of shape (height, width).
    :param xy_boxes: The x,y bounding boxes.
    :param xy_border: A border added to the x,y bounding boxes.
    :param ij_border: A border added to the resulting i,j bounding boxes.
    :param ij_boxes: The resulting i,j bounding boxes.
    """
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
        for j0 in range(h):
            for i0 in range(w):
                x = x_image[j0, i0]
                if x_min <= x <= x_max:
                    y = y_image[j0, i0]
                    if y_min <= y <= y_max:
                        i1 = i0 + 1
                        j1 = j0 + 1
                        i_min = ij_bbox[0]
                        j_min = ij_bbox[1]
                        i_max = ij_bbox[2]
                        j_max = ij_bbox[3]
                        if i_min < 0:
                            ij_bbox[0] = i0
                            ij_bbox[1] = j0
                            ij_bbox[2] = i1
                            ij_bbox[3] = j1
                        else:
                            if i0 < i_min:
                                ij_bbox[0] = i0
                            if j0 < j_min:
                                ij_bbox[1] = j0
                            if i1 > i_max:
                                ij_bbox[2] = i1
                            if j1 > j_max:
                                ij_bbox[3] = j1
        if ij_border != 0 and ij_bbox[0] != -1:
            i_min = ij_bbox[0] - ij_border
            j_min = ij_bbox[1] - ij_border
            i_max = ij_bbox[2] + ij_border
            j_max = ij_bbox[3] + ij_border
            if i_min < 0:
                i_min = 0
            if j_min < 0:
                j_min = 0
            if i_max > w:
                i_max = w
            if j_max > h:
                j_max = h
            ij_bbox[0] = i_min
            ij_bbox[1] = j_min
            ij_bbox[2] = i_max
            ij_bbox[3] = j_max


def compute_xy_bbox(xy_coords: Union[xr.DataArray, np.ndarray, da.Array]) -> Tuple[float, float, float, float]:
    xy_coords = da.asarray(xy_coords)
    result = da.reduction(xy_coords,
                          compute_xy_bbox_chunk,
                          compute_xy_bbox_aggregate,
                          keepdims=True,
                          # concatenate=False,
                          dtype=xy_coords.dtype,
                          axis=(1, 2),
                          meta=np.array([[0, 0], [0, 0]], dtype=xy_coords.dtype))
    x_min, x_max, y_min, y_max = map(float, result.compute().flatten())
    return x_min, y_min, x_max, y_max


@nb.jit(nopython=True)
def compute_xy_bbox_chunk(xy_block: np.ndarray, axis: int, keepdims: bool):
    # print('\ncompute_xy_bbox_chunk:', xy_block, axis, keepdims)
    return compute_xy_bbox_block(xy_block, axis, keepdims)


@nb.jit(nopython=True)
def compute_xy_bbox_aggregate(xy_block: np.ndarray, axis: int, keepdims: bool):
    # print('\ncompute_xy_bbox_aggregate:', xy_block, axis, keepdims)
    return compute_xy_bbox_block(xy_block, axis, keepdims)


@nb.jit(nopython=True)
def compute_xy_bbox_block(xy_block: np.ndarray, axis: int, keepdims: bool):
    x_block = xy_block[0].flatten()
    y_block = xy_block[1].flatten()
    x_min = np.inf
    y_min = np.inf
    x_max = -np.inf
    y_max = -np.inf
    n = x_block.size
    for i in range(n):
        x = x_block[i]
        y = y_block[i]
        if x < x_min:
            x_min = x
        if x > x_max:
            x_max = x
        if y < y_min:
            y_min = y
        if y > y_max:
            y_max = y
    x_min = x_min if x_min != np.inf else np.nan
    y_min = y_min if y_min != np.inf else np.nan
    x_max = x_max if x_max != -np.inf else np.nan
    y_max = y_max if y_max != -np.inf else np.nan
    return np.array([[[x_min, x_max]], [[y_min, y_max]]], dtype=xy_block.dtype)
