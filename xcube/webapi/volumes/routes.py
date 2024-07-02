# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import functools
import gzip
import math
import sys

import numpy as np
import pandas as pd
import pyproj

from xcube.core.gridmapping import CRS_CRS84
from xcube.core.select import select_subset
from xcube.core.varexpr import VarExprContext
from xcube.core.varexpr import split_var_assignment
from xcube.server.api import ApiError
from xcube.server.api import ApiHandler
from .api import api
from .config import DEFAULT_MAX_VOXEL_COUNT
from .context import VolumesContext
from ..datasets import PATH_PARAM_DATASET_ID
from ..datasets import PATH_PARAM_VAR_NAME


# noinspection PyPep8Naming
@api.route("/volumes/{datasetId}/{varName}")
class VolumesContextHandler(ApiHandler[VolumesContext]):
    @api.operation(
        operation_id="getVolume",
        summary="Get the volume data for a variable.",
        description=(
            "This is an experimental feature."
            " The volume data is returned in NRRD format."
            " Details of the NRRD format can be found at"
            " https://teem.sourceforge.net/nrrd/format.html"
        ),
        parameters=[
            PATH_PARAM_DATASET_ID,
            PATH_PARAM_VAR_NAME,
            {
                "name": "bbox",
                "in": "query",
                "description": "Bounding box in degrees (WGS-84)"
                " using format x1,y1,x2,y2",
                "schema": {
                    "type": "string",
                },
            },
            {
                "name": "startDate",
                "in": "query",
                "description": "Start date",
                "schema": {"type": "string", "format": "datetime"},
            },
            {
                "name": "endDate",
                "in": "query",
                "description": "End date",
                "schema": {"type": "string", "format": "datetime"},
            },
            {
                "name": "encoding",
                "in": "query",
                "description": "Encoding of the result",
                "schema": {
                    "type": "string",
                    "enum": ["raw", "gz"],
                    "default": "gz",
                },
            },
        ],
    )
    async def get(self, datasetId: str, varName: str):
        bbox = self.request.get_query_arg("bbox", type=str, default="")
        if bbox:
            try:
                x1, y1, x2, y2 = map(float, [p.strip() for p in bbox.split(",")])
            except (ValueError, TypeError):
                raise ApiError.BadRequest("Invalid bbox")
            bbox = x1, y1, x2, y2
        else:
            bbox = None

        start_date = self.request.get_query_arg(
            "startDate", type=pd.Timestamp, default=None
        )
        end_date = self.request.get_query_arg(
            "endDate", type=pd.Timestamp, default=None
        )

        time_range = [start_date, end_date] if start_date or end_date else None

        encoding = self.request.get_query_arg("encoding", type=str, default="gz")

        if encoding not in ("gz", "raw"):
            raise ApiError.BadRequest('Encoding must be one of "gz" or "raw"')

        ml_dataset = self.ctx.datasets_ctx.get_ml_dataset(datasetId)
        grid_mapping = ml_dataset.grid_mapping

        if bbox is not None and not grid_mapping.crs.is_geographic:
            transformer = pyproj.Transformer.from_crs(CRS_CRS84, grid_mapping.crs)
            x1, y1, x2, y2 = bbox
            (x1, x2), (y1, y2) = transformer.transform((x1, x2), (y1, y2))
            bbox = x1, y1, x2, y2

        var_name, var_expr = split_var_assignment(varName)
        if var_expr:
            dataset = self.ctx.datasets_ctx.get_dataset(datasetId).copy()
            dataset[var_name] = VarExprContext(dataset).evaluate(var_expr)
        else:
            dataset = self.ctx.datasets_ctx.get_dataset(
                datasetId, expected_var_names=[var_name]
            )

        var = select_subset(
            dataset,
            var_names=[var_name],
            time_range=time_range,
            bbox=bbox,
            grid_mapping=ml_dataset.grid_mapping,
        )[var_name]

        if var.ndim != 3:
            raise ApiError.BadRequest(f"Variable must be 3-D, got {var.ndim}-D")

        voxel_count = functools.reduce(lambda x, y: x * y, var.shape, 1)
        max_voxel_count = self.ctx.config.get("VolumesAccess", {}).get(
            "MaxVoxelCount", DEFAULT_MAX_VOXEL_COUNT
        )
        if voxel_count > max_voxel_count:
            raise ApiError.BadRequest(
                f"Volume too large, please select a smaller dataset subset."
                f" Maximum is {max_voxel_count} voxels,"
                f' got {" x ".join(map(str, var.shape))} = {voxel_count}.'
            )

        # TODO (forman): allow for any dtype
        values = var.astype(dtype=np.float32).values
        if not ml_dataset.grid_mapping.is_j_axis_up:
            values = values[:, ::-1, :]
        values = np.where(np.isnan(values), 0.0, values)
        data = values.tobytes(order="C")
        if encoding == "gz":
            data = gzip.compress(data)

        block_size = 1024 * 1024
        num_blocks = math.ceil(len(data) / block_size)

        size_z, size_y, size_x = var.shape
        # TODO (forman): find more suitable normalisation
        scale_x = scale_y = 100.0 / max(size_x, size_y)
        scale_z = 100.0 / size_z

        nrrd_header = (
            "NRRD0004\n"
            "# NRRD 4 Format\n"
            "# see http://teem.sourceforge.net/nrrd/format.html\n"
            "type: float\n"  # TODO (forman): allow for any dtype
            "dimension: 3\n"
            "sizes:"
            f" {size_x} {size_y} {size_z}\n"
            f"encoding:"
            f" {encoding}\n"
            "endian:"
            f" {sys.byteorder}\n"
            "space directions:"
            f" ({scale_x},0,0) (0,{scale_y},0) (0,0,{scale_z})\n"
            "space origin: (0,0,0)\n"
            "\n"
        )

        self.response.set_header("Content-Type", "application/octet-stream")
        self.response.set_header("Cache-Control", "max-age=1")
        self.response.write(bytes(nrrd_header, "utf-8"))
        import time

        for i in range(num_blocks):
            self.response.write(data[i * block_size : i * block_size + block_size])
            time.sleep(1 / 10000)

        await self.response.finish()
