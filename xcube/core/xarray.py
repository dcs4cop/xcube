import threading
from typing import Dict, List, Mapping, Any, Union, Sequence, Optional
from deprecated import deprecated
import numpy as np
import pandas as pd
import xarray as xr

from xcube.core.chunk import chunk_dataset
from xcube.core.dsio import open_cube, write_cube
from xcube.core.dump import dump_dataset
from xcube.core.extract import DEFAULT_INDEX_NAME_PATTERN
from xcube.core.extract import get_cube_point_indexes
from xcube.core.extract import get_cube_values_for_indexes
from xcube.core.extract import get_cube_values_for_points
from xcube.core.extract import get_dataset_indexes
from xcube.core.gridmapping import GridMapping
from xcube.core.level import compute_levels
from xcube.core.new import new_cube
from xcube.core.normalize import DatasetIsNotACubeError
from xcube.core.normalize import decode_cube
from xcube.core.schema import CubeSchema, get_cube_schema
from xcube.core.select import select_variables_subset
from xcube.core.vars2dim import vars_to_dim
from xcube.core.verify import verify_cube


@xr.register_dataset_accessor("xcube")
class DatasetAccessor:
    """The xcube xarray API.

    The API is made available via the `xcube` attribute of
    xarray.Dataset instances.

    It defines new xcube-specific properties for xarray datasets:

    * :attr:cube The subset of variables of this dataset
        which all have cube dimensions (time, ..., <y_name>, <x_name>).
        May be an empty dataset.
    * :attr:non_cube The subset of variables of this dataset
        minus the data variables from :attr:cube.
        May be the same as this dataset.
    * :attr:gm The grid mapping used by this dataset.
        It is an instance of `GridMapping`.
        May be None, if this dataset does not define a grid mapping.
    """

    def __init__(self, dataset: xr.Dataset):
        self._dataset: xr.Dataset = dataset
        self._cube_subset: Optional[xr.Dataset] = None
        self._grid_mapping: Optional[GridMapping] = None
        self._lock = threading.RLock()

    @property
    def cube(self) -> xr.Dataset:
        if self._cube_subset is None:
            with self._lock:
                self._init_cube_subset()
        return self._cube_subset

    @property
    def non_cube(self) -> xr.Dataset:
        if self._cube_subset is None:
            with self._lock:
                self._init_cube_subset()
        return self._non_cube_subset

    @property
    def gm(self) -> Optional[GridMapping]:
        if self._cube_subset is None:
            with self._lock:
                self._init_cube_subset()
        return self._grid_mapping

    def _init_cube_subset(self):
        try:
            cube, grid_mapping, non_cube = decode_cube(
                self._dataset, normalize=True, force_copy=True
            )
        except DatasetIsNotACubeError:
            cube, grid_mapping, non_cube = xr.Dataset(), None, self._dataset
        self._cube_subset = cube
        self._grid_mapping = grid_mapping
        self._non_cube_subset = non_cube

    ########################################################################
    # Old API from here on.
    #
    # Let's quickly agree, if we should deprecate all this stuff. I guess,
    # no one uses it.
    #
    # We should only add props and methods to this accessor
    # that require a certain state to be hold.
    # Such state could be props that are expensive to recompute,
    # such as grid mappings.
    #
    # It causes too much overhead and maintenance work
    # if we continue putting any xcube function here.
    ########################################################################

    @classmethod
    def new(cls, **kwargs) -> xr.Dataset:
        """Create a new empty cube. Useful for testing.

        Refer to `xcube.core.new.new_cube()` for details.
        """
        return new_cube(**kwargs)

    @classmethod
    def open(cls, input_path: str, format_name: str = None, **kwargs) -> xr.Dataset:
        """The `read` method as context manager that auto-closes the data cube read.

        Args:
            input_path: input path
            format_name: format, e.g. "zarr" or "netcdf4"
            kwargs: format-specific keyword arguments

        Returns:
            dataset object
        """
        return open_cube(input_path, format_name=format_name, **kwargs)

    def write(self, output_path: str, format_name: str = None, **kwargs) -> xr.Dataset:
        """Write this cube to *output_path*.
        If *format* is not provided it will be guessed from *output_path*.

        Args:
            output_path: output path
            format_name: format, e.g. "zarr" or "netcdf4"
            **kwargs: format-specific keyword arguments

        Returns:
            the input dataset
        """
        return write_cube(self._dataset, output_path, format_name=format_name, **kwargs)

    def values_for_points(
        self,
        points: Union[xr.Dataset, pd.DataFrame, Mapping[str, Any]],
        var_names: Sequence[str] = None,
        index_name_pattern: str = DEFAULT_INDEX_NAME_PATTERN,
        include_indexes: bool = False,
        method: str = "nearest",
        cube_asserted: bool = False,
    ):
        """Extract values from cube variables at given coordinates in *points*.

        Args:
            points: Dictionary that maps dimension name to coordinate
                arrays.
            var_names: An optional list of names of data variables in
                *cube* whose values shall be extracted.
            index_name_pattern: A naming pattern for the computed
                indexes columns. Must include "{name}" which will be
                replaced by the dimension name.
            include_indexes: Whether to include computed indexes in
                return value.
            method: "nearest" or "linear".
            cube_asserted: If False, *cube* will be verified, otherwise
                it is expected to be a valid cube.

        Returns:
            A new data frame whose columns are values from *cube*
            variables at given *points*.
        """
        return get_cube_values_for_points(
            self._dataset,
            points,
            var_names=var_names,
            index_name_pattern=index_name_pattern,
            include_indexes=include_indexes,
            method=method,
            cube_asserted=cube_asserted,
        )

    def values_for_indexes(
        self,
        indexes: Union[xr.Dataset, pd.DataFrame, Mapping[str, Any]],
        var_names: Sequence[str] = None,
        index_name_pattern: str = DEFAULT_INDEX_NAME_PATTERN,
        method: str = "nearest",
        cube_asserted: bool = False,
    ) -> xr.Dataset:
        """Get values from this cube at given *indexes*.

        Args:
            indexes: A mapping from column names to index and fraction
                arrays for all cube dimensions.
            var_names: An optional list of names of data variables in
                *cube* whose values shall be extracted.
            index_name_pattern: A naming pattern for the computed
                indexes columns. Must include "{name}" which will be
                replaced by the dimension name.
            method: "nearest" or "linear".
            cube_asserted: If False, *cube* will be verified, otherwise
                it is expected to be a valid cube.

        Returns:
            A new data frame whose columns are values from *cube*
            variables at given *indexes*.
        """
        return get_cube_values_for_indexes(
            self._dataset,
            indexes,
            data_var_names=var_names,
            index_name_pattern=index_name_pattern,
            method=method,
            cube_asserted=cube_asserted,
        )

    def point_indexes(
        self,
        points: Union[xr.Dataset, pd.DataFrame, Mapping[str, Any]],
        dim_name_mapping: Mapping[str, str] = None,
        index_name_pattern: str = DEFAULT_INDEX_NAME_PATTERN,
        index_dtype=np.float64,
        cube_asserted: bool = False,
    ):
        """Get indexes of given coordinates in *points* into this cube.

        Args:
            points: A mapping from column names to column data arrays,
                which must all have the same length.
            dim_name_mapping: A mapping from dimension names in *cube*
                to column names in *points*.
            index_name_pattern: A naming pattern for the computed
                indexes columns. Must include "{name}" which will be
                replaced by the dimension name.
            index_dtype: Numpy data type for the indexes. If it is a
                floating point type (default), then *indexes* will
                contain fractions, which may be used for interpolation.
                For out-of-range coordinates in *points*, indexes will
                be -1 if *index_dtype* is an integer type, and NaN, if
                *index_dtype* is a floating point types.
            cube_asserted: If False, *cube* will be verified, otherwise
                it is expected to be a valid cube.

        Returns:
            A dataset containing the index columns.
        """
        return get_cube_point_indexes(
            self._dataset,
            points,
            dim_name_mapping=dim_name_mapping,
            index_name_pattern=index_name_pattern,
            index_dtype=index_dtype,
            cube_asserted=cube_asserted,
        )

    def indexes(
        self,
        coord_var_name: str,
        coord_values: Union[xr.DataArray, np.ndarray],
        index_dtype=np.float64,
    ) -> Union[xr.DataArray, np.ndarray]:
        """Compute the indexes into a coordinate variable *coord_var_name* of this cube
        for the given coordinate values *coord_values*.

        The coordinate variable's labels must be monotonic increasing or decreasing,
        otherwise the result will be nonsense.

        For any value in *coord_values* that is out of the bounds of the coordinate variable's values,
        the index depends on the value of *index_dtype*. If *index_dtype* is an integer type, invalid indexes are
        encoded as -1 while for floating point types, NaN will be used.

        Returns the indexes as an array-like object of type *dtype*.

        Args:
            coord_var_name: Name of a coordinate variable.
            coord_values: Array-like coordinate values.
            index_dtype: Numpy data type for the indexes. If it is a
                floating point type (default), then *indexes* will
                contain fractions, which may be used for interpolation.
                For out-of-range coordinates in *points*, indexes will
                be -1 if *index_dtype* is an integer type, and NaN, if
                *index_dtype* is a floating point types.

        Returns:
            The indexes and their fractions as a tuple of numpy int64
            and float64 arrays.
        """
        return get_dataset_indexes(
            self._dataset,
            coord_var_name=coord_var_name,
            coord_values=coord_values,
            index_dtype=index_dtype,
        )

    def chunk(
        self, chunk_sizes: Dict[str, int] = None, format_name: str = None
    ) -> xr.Dataset:
        """Chunk this dataset and update encodings for given format.

        Args:
            chunk_sizes: mapping from dimension name to new chunk size
            format_name: format, e.g. "zarr" or "netcdf4"

        Returns:
            the re-chunked dataset
        """
        return chunk_dataset(
            self._dataset, chunk_sizes=chunk_sizes, format_name=format_name
        )

    def vars_to_dim(self, dim_name: str = "newdim"):
        """Convert data variables into a dimension

        Args:
            dim_name: The name of the new dimension ['vars']

        Returns:
            A new xcube dataset with the new dimension.
        """
        return vars_to_dim(self._dataset, dim_name)

    def dump(self, var_names=None, show_var_encoding=False) -> str:
        """Dump this dataset or its variables into a text string.

        Args:
            var_names: names of variables to be dumped
            show_var_encoding: also dump variable encodings?

        Returns:
            the dataset dump
        """
        return dump_dataset(
            self._dataset, var_names=var_names, show_var_encoding=show_var_encoding
        )

    def verify(self) -> List[str]:
        """Verify that this dataset is a valid xcube dataset.

        Returns a list of issues, which is empty if this dataset is a valid xcube dataset.

        Returns:
            List of issues or empty list.
        """
        return verify_cube(self._dataset)

    def select_variables_subset(self, var_names: Sequence[str] = None):
        """Select data variable from given *dataset* and create new dataset.

        Args:
            var_names: The names of data variables to select.

        Returns:
            A new dataset. It is empty, if *var_names* is empty. It is
            *dataset*, if *var_names* is None.
        """
        return select_variables_subset(self._dataset, var_names)

    @deprecated(
        version="0.13.0",
        reason="multi-level datasets should be represented by"
        " class xcube.core.mldataset.MultiLevelDataset",
    )
    def levels(self, **kwargs) -> List[xr.Dataset]:
        """Transform this dataset into the levels of a multi-level pyramid with spatial resolution
        decreasing by a factor of two in both spatial dimensions.

        It is assumed that the spatial dimensions of each variable are the inner-most, that is, the last two elements
        of a variable's shape provide the spatial dimension sizes.

        Args:
            spatial_dims: If given, only variables are considered whose
                last to dimension elements match the given
                *spatial_dims*.
            spatial_shape: If given, only variables are considered whose
                last to shape elements match the given *spatial_shape*.
            spatial_tile_shape: If given, chunking will match the
                provided *spatial_tile_shape*.
            var_names: Variables to consider. If None, all variables
                with at least two dimensions are considered.
            max_num_levels: If given, the maximum number of pyramid
                levels.
            post_process_level: If given, the function will be called
                for each level and must return a dataset.
            progress_monitor: If given, the function will be called for
                each level.

        Returns:
            A list of dataset instances representing the multi-level
            pyramid.
        """
        return compute_levels(self._dataset, **kwargs)

    @property
    def schema(self) -> CubeSchema:
        return get_cube_schema(self._dataset)
