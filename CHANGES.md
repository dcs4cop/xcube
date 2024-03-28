## Changes in 1.4.2 (in development)

* Update dependencies to better match imports; remove the defaults channel;
  turn adlfs into a soft dependency. (#945)

* Reformatted xcube code base using [black](https://black.readthedocs.io/)
  default settings. It implies a line length of 88 characters and double quotes 
  for string literals. Also added [`.editorconfig`](https://editorconfig.org/) 
  for IDEs not recognising black's defaults.

* Renamed xcube's main branch from `master` to `main` on GitHub.

* xcube's code base changed its docstring format from reST style to the much better 
  readable [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).
  Existing docstrings have been converted using the awesome [docconvert](https://github.com/cbillingham/docconvert) 
  tool.

* Added documentation page "How do I ..." that points users to applicable
  xcube Python API.

## Changes in 1.4.1

### Enhancements

* Data stores can now return _data iterators_ from their `open_data()` method.
  For example, a data store implementation can now return a data cube either
  with a time dimension of size 100, or could be asked to return 100 cube
  time slices with dimension size 1 in form of an iterator.
  This feature has been added to effectively support the new
  [zappend](https://github.com/bcdev/zappend) tool. (#919)

### Fixes

* Fix two OGC Collections unit tests that were failing under Windows. (#937)

### Other changes

* Minor updates to make xcube compatible with pandas 2 and Python 3.12. (#933)

* Minor updates to make xcube compatible with xarray >=2023.9.0. (#897, #939)

## Changes in 1.4.0

### Enhancements

* Added new `reference` filesystem data store to support 
  "kerchunked" NetCDF files in object storage. (#928)
  
  See also
    - [ReferenceFileSystem](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.implementations.reference.ReferenceFileSystem)
    - [kerchunk](https://github.com/fsspec/kerchunk)

* Improved xcube Server's STAC API:
  * Provide links for multiple coverages data formats
  * Add `crs` and `crs_storage` properties to STAC data
  * Add spatial and temporal grid data to collection descriptions
  * Add a schema endpoint returning a JSON schema of a dataset's data
    variables
  * Add links to domain set, range type, and range schema to collection
    descriptions

* Improved xcube Server's Coverages API:
  * Support scaling parameters `scale-factor`, `scale-axes`, and `scale-size`
  * Improve handling of bbox parameters
  * Handle half-open datetime intervals
  * More robust and standard-compliant parameter parsing and checking
  * More informative responses for incorrect or unsupported parameters
  * Omit unnecessary dimensions in TIFF and PNG coverages
  * Use crs_wkt when determining CRS, if present and needed
  * Change default subsetting and bbox CRS from EPSG:4326 to OGC:CRS84
  * Implement reprojection for bbox
  * Ensure datetime parameters match dataset’s timezone awareness
  * Reimplement subsetting (better standards conformance, cleaner code)
  * Set Content-Bbox and Content-Crs headers in the HTTP response
  * Support safe CURIE syntax for CRS specification

### Fixes

* Fixed `KeyError: 'lon_bnds'` raised occasionally when opening 
  (mostly NetCDF) datasets. (#930)
* Make S3 unit tests compatible with moto 5 server. (#922)
* Make some CLI unit tests compatible with pytest 8. (#922)
* Rename some test classes to avoid spurious warnings. (#924)

### Other changes

* Require Python >=3.9 (previously >=3.8)

## Changes in 1.3.1

* Updated Dockerfile and GitHub workflows; no changes to the xcube codebase
  itself

## Changes in 1.3.0

### Enhancements

* Added a basic implementation of the draft version of OGC API - Coverages.
  (#879, #889, #900)
* Adapted the STAC implementation to additionally offer datasets as
  individual collections for better integration with OGC API - Coverages.
  (#889)
* Various minor improvements to STAC implementation. (#900)

### Fixes

* Resolved the issue for CRS84 error due to latest version of gdal (#869)
* Fixed incorrect additional variable data in STAC datacube properties. (#889)
* Fixed access of geotiff datasets from public s3 buckets (#893)

### Other changes

* `update_dataset_attrs` can now also handle datasets with CRS other than 
  WGS84 and update the metadata according to the 
  [ESIP Attribute Convention for Data Discovery](https://wiki.esipfed.org/Attribute_Convention_for_Data_Discovery_1-3#Recommended). 
* removed deprecated module xcube edit, which has been deprecated since 
  version 0.13.0
* Update "Development process" section of developer guide.
* Updated GitHub workflow to build docker image for GitHub releases only and 
  not on each commit to main.

## Changes in 1.2.0

### Enhancements

* Added a new, experimental `/compute` API to xcube server. 
  It comprises the following endpoints:
  - `GET compute/operations` - List available operations.
  - `GET compute/operations/{opId}` - Get details of a given operation.
  - `PUT compute/jobs` - Start a new job that executes an operation.
  - `GET compute/jobs` - Get all jobs.
  - `GET compute/jobs/{jobId}` - Get details of a given job.
  - `DELETE compute/jobs/{jobId}` - Cancel a given job.
  
  The available operations are currently taken from module
  `xcube.webapi.compute.operations`.
  
  To disable the new API use the following server configuration:
  ```yaml
  api_spec:
    excludes: ["compute"] 
  ...
  ```

### Other changes

* Added `shutdown_on_close=True` parameter to coiled params to ensure that the 
  clusters are shut down on close. (#881)
* Introduced new parameter `region` for utility function `new_cluster` in 
  `xcube.util.dask` which will ensure coiled creates the dask cluster in the 
  prefered default region: eu-central-1. (#882)
* Server offers the function `add_place_group` in `places/context.py`,
  which allows plugins to add place groups from external sources.
  

## Changes in 1.1.3

### Fixes

* Fixed Windows-only bug in `xcube serve --config <path>`: 
  If config `path` is provided with back-slashes, a missing `base_dir` 
  config parameter is now correctly set to the parent directory of `path`. 
  Before, the current working directory was used.

### Other changes

* Updated AppVeyor and GitHub workflow configurations to use micromamba rather
  than mamba (#785)

## Changes in 1.1.2

### Fixes

* Fixed issue where geotiff access from a protected s3 bucket was denied (#863)

## Changes in 1.1.1

* Bundled new build of [xcube-viewer 1.1.0.1](https://github.com/dcs4cop/xcube-viewer/releases/tag/v1.1.0)
  that will correctly respect a given xcube server from loaded from the 
  viewer configuration.

## Changes in 1.1.0

### Enhancements

* Bundled [xcube-viewer 1.1.0](https://github.com/dcs4cop/xcube-viewer/releases/tag/v1.1.0).

* Updated installation instructions (#859)

* Included support for FTP filesystem by adding a new data store `ftp`. 

  These changes will enable access to data cubes (`.zarr` or `.levels`) 
  in FTP storage as shown here: 
  
  ```python
  store = new_data_store(
      "ftp",                     # FTP filesystem protocol
      root="path/to/files",      # Path on FTP server
      storage_options= {'host':  'ftp.xxx',  # The url to the ftp server
                        'port': 21           # Port, defaults to 21  
                        # Optionally, use 
                        # 'username': 'xxx'
                        # 'password': 'xxx'}  
  )
  store.list_data_ids()
  ```
  Note that there is no anon parameter, as the store will assume no anonymity
  if no username and password are set.
  
  Same configuration for xcube Server:

  ```yaml
  DataStores:
  - Identifier: siec
    StoreId: ftp
    StoreParams:
      root: my_path_on_the_host
      max_depth: 1
      storage_options:
        host: "ftp.xxx"
        port: xxx
        username: "xxx"
        password': "xxx"
  ``` 

* Updated [xcube Dataset Specification](docs/source/cubespec.md).
  (addressing #844)

* Added [xcube Data Access](docs/source/dataaccess.md) documentation.

### Fixes 

* Fixed various issues with the auto-generated Python API documentation.

* Fixed a problem where time series requests may have missed outer values
  of a requested time range. (#860)
  - Introduced query parameter `tolerance` for
    endpoint `/timeseries/{datasetId}/{varName}` which is
    the number of seconds by which the given time range is expanded. Its 
    default value is one second to overcome rounding problems with 
    microsecond fractions. (#860)
  - We now round the time dimension labels for a dataset as 
    follows (rounding frequency is 1 second by default):
    - First times stamp: `floor(time[0])`
    - Last times stamp: `ceil(time[-1])`
    - In-between time stamps: `round(time[1: -1])`

### Other changes

* Pinned `gdal` dependency to `>=3.0, <3.6.3` due to incompatibilities.

## Changes in 1.0.5

* When running xcube in a JupyterLab, the class
  `xcube.webapi.viewer.Viewer` can be used to programmatically 
  launch a xcube Viewer UI. 
  The class now recognizes an environment variable `XCUBE_JUPYTER_LAB_URL` 
  that contains a JupyterLab's public base URL for a given user. 
  To work properly, the 
  [jupyter-server-proxy](https://jupyter-server-proxy.readthedocs.io/) 
  extension must be installed and enabled.

* Bundled [xcube-viewer 1.0.2.1](https://github.com/dcs4cop/xcube-viewer/releases/tag/v1.0.2.1).

## Changes in 1.0.4 

* Setting a dataset's `BoundingBox` in the server configuration 
  is now recognised when requesting the dataset details. (#845)

* It is now possible to enforce the order of variables reported by 
  xcube server. The new server configuration key `Variables` can be added 
  to `Datasets` configurations. Is a list of wildcard patterns that 
  determines the order of variables and the subset of variables to be 
  reported. (#835) 

* Pinned Pandas dependency to lower than 2.0 because of incompatibility 
  with both xarray and xcube 
  (see https://github.com/pydata/xarray/issues/7716). 
  Therefore, the following xcube deprecations have been introduced:
  - The optional `--base/-b` of the `xcube resample` CLI tool.
  - The keyword argument `base` of the  `xcube.core.resample.resample_in_time` 
    function.

* Bundled [xcube-viewer 1.0.2](https://github.com/dcs4cop/xcube-viewer/releases/tag/v1.0.2).

## Changes in 1.0.3

Same as 1.0.2, just fixed unit tests due to minor Python environment change.

## Changes in 1.0.2

* Bundled latest 
  [xcube-viewer 1.0.1](https://github.com/dcs4cop/xcube-viewer/releases/tag/v1.0.1).

* xcube is now compatible with Python 3.10. (#583)

* The `Viewer.add_dataset()` method of the xcube JupyterLab integration 
  has been enhanced by two optional keyword arguments `style` and 
  `color_mappings` to allow for customized, initial color mapping
  of dataset variables. The example notebook 
  [xcube-viewer-in-jl.ipynb](examples/notebooks/viewer/xcube-viewer-in-jl.ipynb)
  has been updated to reflect the enhancement.

* Fixed an issue with new xcube data store `abfs` 
  for the Azure Blob filesystem. (#798)

## Changes in 1.0.1

### Fixes

* Fixed recurring issue where xcube server was unable to locate Python
  code downloaded from S3 when configuring dynamically computed datasets
  (configuration `FileSystem: memory`) or augmenting existing datasets 
  by dynamically computed variables (configuration `Augmentation`). (#828)


## Changes in 1.0.0 

### Enhancements

* Added a catalog API compliant to [STAC](https://stacspec.org/en/) to 
  xcube server. (#455)
  - It serves a single collection named "datacubes" whose items are the
    datasets published by the service. 
  - The collection items make use the STAC 
    [datacube](https://github.com/stac-extensions/datacube) extension. 

* Simplified the cloud deployment of xcube server/viewer applications (#815). 
  This has been achieved by the following new xcube server features:
  - Configuration files can now also be URLs which allows 
    provisioning from S3-compatible object storage. 
    For example, it is now possible to invoke xcube server as follows: 
    ```bash
    $ xcube serve --config s3://cyanoalert/xcube/demo.yaml ...
    ```
  - A new endpoint `/viewer/config/{*path}` allows 
    for configuring the viewer accessible via endpoint `/viewer`. 
    The actual source for the configuration items is configured by xcube 
    server configuration using the new entry `Viewer/Configuration/Path`, 
    for example:
    ```yaml
    Viewer:
      Configuration:
        Path: s3://cyanoalert/xcube/viewer-config
    ```
  - A typical xcube server configuration comprises many paths, and 
    relative paths of known configuration parameters are resolved against 
    the `base_dir` configuration parameter. However, for values of 
    parameters passed to user functions that represent paths in user code, 
    this cannot be done automatically. For such situations, expressions 
    can be used. An expression is any string between `"${"` and `"}"` in a 
    configuration value. An expression can contain the variables
    `base_dir` (a string), `ctx` the current server context 
    (type `xcube.webapi.datasets.DatasetsContext`), as well as the function
    `resolve_config_path(path)` that is used to make a path absolut with 
    respect to `base_dir` and to normalize it. For example
    ```yaml
    Augmentation:
      Path: augmentation/metadata.py
      Function: metadata:update_metadata
      InputParameters:
        bands_config: ${resolve_config_path("../common/bands.yaml")}
    ```

* xcube's spatial resampling functions `resample_in_space()`,
  `affine_transform_dataset()`,  and `rectify_dataset()` exported 
  from module `xcube.core.resampling` now encode the target grid mapping 
  into the resampled datasets. (#822) 
  
  This new default behaviour can be switched off by keyword argument 
  `encode_cf=False`. 
  The grid mapping name can be set by keyword argument `gm_name`. 
  If `gm_name` is not given a grid mapping will not be encoded if 
  all the following conditions are true: 
  - The target CRS is geographic; 
  - The spatial dimension names are "lon" and "lat";
  - The spatial 1-D coordinate variables are named "lon" and "lat" 
    and are evenly spaced.  

  The encoding of the grid mapping is done according to CF conventions:
  - The CRS is encoded as attributes of a 0-D data variable named by `gm_name`
  - All spatial data variables receive an attribute `grid_mapping` that is
    set to the value of `gm_name`.
  
* Added Notebook 
  [xcube-viewer-in-jl.ipynb](examples/notebooks/viewer/xcube-viewer-in-jl.ipynb)
  that explains how xcube Viewer can now be utilised in JupyterLab
  using the new (still experimental) xcube JupyterLab extension
  [xcube-jl-ext](https://github.com/dcs4cop/xcube-jl-ext).
  The `xcube-jl-ext` package is also available on PyPI.

* Updated example 
  [Notebook for CMEMS data store](examples/notebooks/datastores/7_cmems_data_store.ipynb)
  to reflect changes of parameter names that provide CMEMS API credentials.

* Included support for Azure Blob Storage filesystem by adding a new 
  data store `abfs`. Many thanks to [Ed](https://github.com/edd3x)!
  (#752)

  These changes will enable access to data cubes (`.zarr` or `.levels`) 
  in Azure blob storage as shown here: 
  
  ```python
  store = new_data_store(
      "abfs",                    # Azure filesystem protocol
      root="my_blob_container",  # Azure blob container name
      storage_options= {'anon': True, 
                        # Alternatively, use 'connection_string': 'xxx'
                        'account_name': 'xxx', 
                        'account_key':'xxx'}  
  )
  store.list_data_ids()
  ```
  
  Same configuration for xcube Server:

  ```yaml
  DataStores:
  - Identifier: siec
    StoreId: abfs
    StoreParams:
      root: my_blob_container
      max_depth: 1
      storage_options:
        anon: true
        account_name: "xxx"
        account_key': "xxx"
        # or
        # connection_string: "xxx"
    Datasets:
      - Path: "*.levels"
        Style: default
  ```
  
* Added Notebook
  [8_azure_blob_filesystem.ipynb](examples/notebooks/datastores/8_azure_blob_filesystem.ipynb). 
  This notebook shows how a new data store instance can connect and list 
  Zarr files from Azure bolb storage using the new `abfs` data store. 

* xcube's `Dockerfile` no longer creates a conda environment `xcube`.
  All dependencies are now installed into the `base` environment making it 
  easier to use the container as an executable for xcube applications.
  We are now also using a `micromamba` base image instead of `miniconda`.
  The result is a much faster build and smaller image size.

* Added a `new_cluster` function to `xcube.util.dask`, which can create
  Dask clusters with various configuration options.

* The xcube multi-level dataset specification has been enhanced. (#802)
  - When writing multi-level datasets (`*.levels/`) we now create a new 
    JSON file `.zlevels` that contains the parameters used to create the 
    dataset.
  - A new class `xcube.core.mldataset.FsMultiLevelDataset` that represents
    a multi-level dataset persisted to some filesystem, like 
    "file", "s3", "memory". It can also write datasets to the filesystem. 


* Changed the behaviour of the class 
  `xcube.core.mldataset.CombinedMultiLevelDataset` to do what we 
  actually expect:
  If the keyword argument `combiner_func` is not given or `None` is passed, 
  a copy of the first dataset is made, which is then subsequently updated 
  by the remaining datasets using `xarray.Dataset.update()`.
  The former default was using the `xarray.merge()`, which for some reason
  can eagerly load Dask array chunks into memory that won't be released. 

### Fixes

* Tiles of datasets with forward slashes in their identifiers
  (originated from nested directories) now display again correctly
  in xcube Viewer. Tile URLs have not been URL-encoded in such cases. (#817)

* The xcube server configuration parameters `url_prefix` and 
  `reverse_url_prefix` can now be absolute URLs. This fixes a problem for 
  relative prefixes such as `"proxy/8000"` used for xcube server running 
  inside JupyterLab. Here, the expected returned self-referencing URL was
  `https://{host}/users/{user}/proxy/8000/{path}` but we got
  `http://{host}/proxy/8000/{path}`. (#806)

## Changes in 0.13.0

### Enhancements

* xcube Server has been rewritten almost from scratch.

  - Introduced a new endpoint `${server_url}/s3` that emulates
    and AWS S3 object storage for the published datasets. (#717)
    The `bucket` name can be either:
    * `s3://datasets` - publishes all datasets in Zarr format.
    * `s3://pyramids` - publishes all datasets in a multi-level `levels`
      format (multi-resolution N-D images)
      that comprises level datasets in Zarr format.
    
    Datasets published through the S3 API are slightly 
    renamed for clarity. For bucket `s3://pyramids`:
    * if a dataset identifier has suffix `.levels`, the identifier remains;
    * if a dataset identifier has suffix `.zarr`, it will be replaced by 
      `.levels` only if such a dataset doesn't exist;
    * otherwise, the suffix `.levels` is appended to the identifier.
    For bucket `s3://datasets` the opposite is true:
    * if a dataset identifier has suffix `.zarr`, the identifier remains;
    * if a dataset identifier has suffix `.levels`, it will be replaced by 
      `.zarr` only if such a dataset doesn't exist;
    * otherwise, the suffix `.zarr` is appended to the identifier.

    With the new S3 endpoints in place, xcube Server instances can be used
    as xcube data stores as follows:
    
    ```python
    store = new_data_store(
        "s3", 
        root="datasets",   # bucket "datasets", use also "pyramids"
        max_depth=2,       # optional, but we may have nested datasets
        storage_options=dict(
            anon=True,
            client_kwargs=dict(
                endpoint_url='http://localhost:8080/s3' 
            )
        )
    )
    ```

  - The limited `s3bucket` endpoints are no longer available and are 
    replaced by `s3` endpoints. 

  - Added new endpoint `/viewer` that serves a self-contained, 
    packaged build of 
    [xcube Viewer](https://github.com/dcs4cop/xcube-viewer). 
    The packaged viewer can be overridden by environment variable 
    `XCUBE_VIEWER_PATH` that must point to a directory with a 
    build of a compatible viewer.

  - The `--show` option of `xcube serve` 
    has been renamed to `--open-viewer`. 
    It now uses the self-contained, packaged build of 
    [xcube Viewer](https://github.com/dcs4cop/xcube-viewer). (#750)

  - The `--show` option of `xcube serve` 
    now outputs various aspects of the server configuration. 
  
  - Added experimental endpoint `/volumes`.
    It is used by xcube Viewer to render 3-D volumes.

* xcube Server is now more tolerant with respect to datasets it can not 
  open without errors. Implementation detail: It no longer fails if 
  opening datasets raises any exception other than `DatasetIsNotACubeError`.
  (#789)

* xcube Server's colormap management has been improved in several ways:
  - Colormaps are no longer managed globally. E.g., on server configuration 
    change, new custom colormaps are reloaded from files. 
  - Colormaps are loaded dynamically from underlying 
    matplotlib and cmocean registries, and custom SNAP color palette files. 
    That means, latest matplotlib colormaps are now always available. (#687)
  - Colormaps can now be reversed (name suffix `"_r"`), 
    can have alpha blending (name suffix `"_alpha"`),
    or both (name suffix `"_r_alpha"`).
  - Loading of custom colormaps from SNAP `*.cpd` has been rewritten.
    Now also the `isLogScaled` property of the colormap is recognized. (#661)
  - The module `xcube.util.cmaps` has been redesigned and now offers
    three new classes for colormap management:
    * `Colormap` - a colormap 
    * `ColormapCategory` - represents a colormap category
    * `ColormapRegistry` - manages colormaps and their categories


* The xcube filesystem data stores such as "file", "s3", "memory"
  can now filter the data identifiers reported by `get_data_ids()`. (#585)
  For this purpose, the data stores now accept two new optional keywords
  which both can take the form of a wildcard pattern or a sequence 
  of wildcard patterns:

  1. `excludes`: if given and if any pattern matches the identifier, 
     the identifier is not reported. 
  2. `includes`: if not given or if any pattern matches the identifier, 
     the identifier is reported.
  
* Added convenience method `DataStore.list_data_ids()` that works 
  like `get_data_ids()`, but returns a list instead of an iterator. (#776)

* Replaced usages of deprecated numpy dtype `numpy.bool` 
  by Python type `bool`. 


### Fixes

* xcube CLI tools no longer emit warnings when trying to import
  installed packages named `xcube_*` as xcube plugins.
  
* The `xcube.util.timeindex` module can now handle 0-dimensional 
  `ndarray`s as indexers. This effectively avoids the warning 
  `Can't determine indexer timezone; leaving it unmodified.`
  which was emitted in such cases.

* `xcube serve` will now also accept datasets with coordinate names
  `longitude` and `latitude`, even if the attribute `long_name` isn't set.
  (#763)

* Function `xcube.core.resampling.affine.affine_transform_dataset()`
  now assumes that geographic coordinate systems are equal by default and
  hence a resampling based on an affine transformation can be performed.

* Fixed a problem with xcube server's WMTS implementation.
  For multi-level resolution datasets with very coarse low resolution levels, 
  the tile matrix sets `WorldCRS84Quad` and `WorldWebMercatorQuad` have 
  reported a negative minimum z-level.

* Implementation of function `xcube.core.geom.rasterize_features()` 
  has been changed to account for consistent use of a target variable's
  `fill_value` and `dtype` for a given feature.
  In-memory (decoded) variables now always use dtype `float64` and use 
  `np.nan` to represent missing values. Persisted (encoded) variable data
  will make use of the target `fill_value` and `dtype`. (#778)

* Relative local filesystem paths to datasets are now correctly resolved 
  against the base directory of the xcube Server's configuration, i.e.
  configuration parameter `base_dir`. (#758)

* Fixed problem with `xcube gen` raising `FileNotFoundError`
  with Zarr >= 2.13.

* Provided backward compatibility with Python 3.8. (#760)

### Other

* The CLI tool `xcube edit` has been deprecated in favour of the 
  `xcube patch`. (#748)

* Deprecated CLI `xcube tile` has been removed.

* Deprecated modules, classes, methods, and functions
  have finally been removed:
  - `xcube.core.geom.get_geometry_mask()`
  - `xcube.core.mldataset.FileStorageMultiLevelDataset`
  - `xcube.core.mldataset.open_ml_dataset()`
  - `xcube.core.mldataset.open_ml_dataset_from_local_fs()`
  - `xcube.core.mldataset.open_ml_dataset_from_object_storage()`
  - `xcube.core.subsampling.get_dataset_subsampling_slices()`
  - `xcube.core.tiledimage`
  - `xcube.core.tilegrid`

* The following classes, methods, and functions have been deprecated:
  - `xcube.core.xarray.DatasetAccessor.levels()`
  - `xcube.util.cmaps.get_cmap()`
  - `xcube.util.cmaps.get_cmaps()`
  
* A new function `compute_tiles()` has been 
  refactored out from function `xcube.core.tile.compute_rgba_tile()`.

* Added method `get_level_for_resolution(xy_res)` to 
  abstract base class `xcube.core.mldataset.MultiLevelDataset`. 

* Removed outdated example resources from `examples/serve/demo`.

* Account for different spatial resolutions in x and y in 
  `xcube.core.geom.get_dataset_bounds()`.

* Make code robust against 0-size coordinates in 
  `xcube.core.update._update_dataset_attrs()`.

* xcube Server has been enhanced to load multi-module Python code 
  for dynamic cubes both from both directories and zip archives.
  For example, the following dataset definition computes a dynamic 
  cube from dataset "local" using function "compute_dataset" in 
  Python module "resample_in_time.py":
  ```yaml
    Path: resample_in_time.py
    Function: compute_dataset
    InputDatasets: ["local"]
  ```
  Users can now pack "resample_in_time.py" among any other modules and 
  packages into a zip archive. Note that the original module name 
  is now a prefix to the function name:
  ```yaml
    Path: modules.zip
    Function: resample_in_time:compute_dataset
    InputDatasets: ["local"]
  ```
  
  Implementation note: this has been achieved by using 
  `xcube.core.byoa.CodeConfig` in
  `xcube.core.mldataset.ComputedMultiLevelDataset`.

* Instead of the `Function` keyword it is now
  possible to use the `Class` keyword.
  While `Function` references a function that receives one or 
  more datasets (type `xarray.Dataset`) and returns a new one, 
  `Class` references a callable that receives one or 
  more multi-level datasets and returns a new one.
  The callable is either a class derived from  
  or a function that returns an instance of 
  `xcube.core.mldataset.MultiLevelDataset`. 

* Module `xcube.core.mldataset` has been refactored into 
  a sub-package for clarity and maintainability.

* Removed deprecated example `examples/tile`.

### Other Changes

* The utility function `xcube.util.dask.create_cluster()` now also
  generates the tag `user` for the current user's name.

## Changes in 0.12.1 

### Enhancements

* Added a new package `xcube.core.zarrstore` that exports a number
  of useful 
  [Zarr store](https://zarr.readthedocs.io/en/stable/api/storage.html) 
  implementations and Zarr store utilities: 
  * `xcube.core.zarrstore.GenericZarrStore` comprises 
    user-defined, generic array definitions. Arrays will compute 
    their chunks either from a function or a static data array. 
  * `xcube.core.zarrstore.LoggingZarrStore` is used to log 
    Zarr store access performance and therefore useful for 
    runtime optimisation and debugging. 
  * `xcube.core.zarrstore.DiagnosticZarrStore` is used for testing
    Zarr store implementations. 
  * Added a xarray dataset accessor 
    `xcube.core.zarrstore.ZarrStoreHolder` that enhances instances of
    `xarray.Dataset` by a new property `zarr_store`. It holds a Zarr store
    instance that represents the datasets as a key-value mapping.
    This will prepare later versions of xcube Server for publishing all 
    datasets via an emulated S3 API.

    In turn, the classes of module `xcube.core.chunkstore` have been
    deprecated.
    
* Added a new function `xcube.core.select.select_label_subset()` that 
  is used to select dataset labels along a given dimension using
  user-defined predicate functions.

* The xcube Python environment is now requiring 
  `xarray >= 2022.6` and `zarr >= 2.11` to ensure sparse 
  Zarr datasets can be written using `dataset.to_zarr(store)`. (#688)

* Added new module `xcube.util.jsonencoder` that offers the class 
  `NumpyJSONEncoder` used to serialize numpy-like scalar values to JSON. 
  It also offers the function `to_json_value()` to convert Python objects 
  into JSON-serializable versions. The new functionality is required 
  to ensure dataset attributes that are JSON-serializable. For example,
  the latest version of the `rioxarray` package generates a `_FillValue` 
  attribute with datatype `np.uint8`. 

### Fixes

* The filesystem-based data stores for the "s3", "file", and "memory"
  protocols can now provide `xr.Dataset` instances from image pyramids
  formats, i.e. the `levels` and `geotiff` formats.

## Changes in 0.12.0

### Enhancements

* Allow xcube Server to work with any OIDC-compliant auth service such as
  Auth0, Keycloak, or Google. Permissions of the form 
  `"read:dataset:\<dataset\>"` and `"read:variable:\<dataset\>"` can now be
  passed by two id token claims: 
  - `permissions` must be a JSON list of permissions;
  - `scope` must be a space-separated character string of permissions.

  It is now also possible to include id token claim values into the 
  permissions as template variables. For example, if the currently
  authenticated user is `demo_user`, the permission 
  `"read:dataset:$username/*"` will effectively be
  `"read:dataset:demo_user/*"` and only allow access to datasets
  with resource identifiers having the prefix `demo_user/`.

  With this change, server configuration has changed:     
  #### Example of OIDC configuration for auth0
  
  Please note, there **must be** a trailing slash in the "Authority" URL.
  
  ```yaml
  Authentication:
    Authority: https://some-demo-service.eu.auth0.com/
    Audience: https://some-demo-service/api/
  ```  
  #### Example of OIDC configuration for Keycloak
  
  Please note, **no** trailing slash in the "Authority" URL.

  ```yaml
  Authentication: 
    Authority: https://kc.some-demo-service.de/auth/realms/some-kc-realm
    Audience: some-kc-realm-xc-api
  ```
* Filesystem-based data stores like "file" and "s3" support reading 
  GeoTIFF and Cloud Optimized GeoTIFF (COG). (#489) 

* `xcube server` now also allows publishing also 2D datasets 
  such as opened from GeoTIFF / COG files.

* Removed all upper version bounds of package dependencies.
  This increases compatibility with existing Python environments.

* A new CLI tool `xcube patch` has been added. It allows for in-place
  metadata patches of Zarr data cubes stored in almost any filesystem 
  supported by [fsspec](https://filesystem-spec.readthedocs.io/en/latest/) 
  including the protocols "s3" and "file". It also allows patching
  xcube multi-level datasets (`*.levels` format).
  
* In the configuration for `xcube server`, datasets defined in `DataStores` 
  may now have user-defined identifiers. In case the path does not unambiguously 
  define a dataset (because it contains wildcards), providing a 
  user-defined identifier will raise an error. 

### Fixes

* xcube Server did not find any grid mapping if a grid mapping variable
  (e.g. spatial_ref or crs) encodes a geographic CRS
  (CF grid mapping name "latitude_longitude") and the related geographical 
  1-D coordinates were named "x" and "y". (#706) 
* Fixed typo in metadata of demo cubes in `examples/serve/demo`. 
  Demo cubes now all have consolidated metadata.
* When writing multi-level datasets with file data stores, i.e.,
  ```python
  store.write_data(dataset, data_id="test.levels", use_saved_levels=True)
  ``` 
  and where `dataset` has different spatial resolutions in x and y, 
  an exception was raised. This is no longer the case. 
* xcube Server can now also compute spatial 2D datasets from users' 
  Python code. In former versions, spatio-temporal 3D cubes were enforced.

### Other important changes

* Deprecated all functions and classes defined in `xcube.core.dsio` 
  in favor of the xcube data store API defined by `xcube.core.store`.

## Changes in 0.11.2

### Enhancements

* `xcube serve` now provides new metadata details of a dataset:
  - The spatial reference is now given by property `spatialRef` 
    and provides a textual representation of the spatial CRS.
  - The dataset boundary is now given as property `geometry`
    and provides a GeoJSON Polygon in geographic coordinates. 
    
* `xcube serve` now publishes the chunk size of a variable's 
  time dimension for either for an associated time-chunked dataset or the
  dataset itself (new variable integer property `timeChunkSize`).
  This helps clients (e.g. xcube Viewer) to improve the 
  server performance for time-series requests.

* The functions
  - `mask_dataset_by_geometry()` 
  - `rasterize_features()`
  of module `xcube.core.geom` have been reimplemented to generate 
  lazy dask arrays. Both should now be applicable to datasets
  that have arbitrarily large spatial dimensions. 
  The spatial chunk sizes to be used can be specified using 
  keyword argument `tile_size`. (#666)

### Fixes

* Fixed ESA CCI example notebook. (#680)

* `xcube serve` now provides datasets after changes of the service 
  configuration while the server is running.
  Previously, it was necessary to restart the server to load the changes. (#678)

### Other changes

* `xcube.core.resampling.affine_transform_dataset()` has a new 
  keyword argument `reuse_coords: bool = False`. If set to `True` 
  the returned dataset will reuse the _same_ spatial coordinates 
  as the target. This is a workaround for xarray issue 
  https://github.com/pydata/xarray/issues/6573.

* Deprecated following functions of module `xcube.core.geom`:
  - `is_dataset_y_axis_inverted()` is no longer used;
  - `get_geometry_mask()` is no longer used;
  - `convert_geometry()` has been renamed to `normalize_geometry()`.
  
## Changes in 0.11.1

* Fixed broken generation of composite RGBA tiles. (#668)
* Fixing broken URLs in xcube viewer documentation, more revision still needed.

## Changes in 0.11.0

### Enhancements

* `xcube serve` can now serve datasets with arbitrary spatial 
  coordinate reference systems. Before xcube 0.11, datasets where forced
  to have a geographical CRS such as EPSG:4326 or CRS84. 

* `xcube serve` can now provide image tiles for two popular tile grids:
  1. global geographic grid, with 2 x 1 tiles at level zero (the default);
  2. global web mercator grid, with 1 x 1 tiles at level 
     zero ("Google projection", OSM tile grid).
  
  The general form of the new xcube tile URL is (currently)
       
      /datasets/{ds_id}/vars/{var_name}/tile2/{z}/{y}/{x}
    
  The following query parameters can be used

  - `crs`: set to `CRS84` to use the geographical grid (the default),
    or `EPSG:3857` to use the web mercator grid. 
  - `cbar`: color bar name such as `viridis` or `plasma`, 
     see color bar names of matplotlib. Defaults to `bone`.
  - `vmin`: minimum value to be used for color mapping. Defaults to `0`.
  - `vmax`: maximum value to be used for color mapping. Defaults to `1`.
  - `retina`: if set to `1`, tile size will be 512 instead of 256.

* The WMTS provided by `xcube serve` has been reimplemented from scratch.
  It now provides two common tile matrix sets:
  1. `WorldCRS84Quad` global geographic grid, with 2 x 1 tiles at level zero; 
  2. `WorldWebMercatorQuad` global web mercator grid, with 1 x 1 tiles 
     at level zero. 
  
  New RESTful endpoints have been added to reflect this:

      /wmts/1.0.0/{TileMatrixSet}/WMTSCapabilities.xml
      /wmts/1.0.0/tile/{Dataset}/{Variable}/{TileMatrixSet}/{TileMatrix}/{TileRow}/{TileCol}.png
  
  The existing RESTful endpoints now use tile matrix set `WorldCRS84Quad` by default:

      /wmts/1.0.0/WMTSCapabilities.xml
      /wmts/1.0.0/tile/{Dataset}/{Variable}/{TileMatrix}/{TileRow}/{TileCol}.png

  The key-value pair (KVP) endpoint `/wmts/kvp` now recognises the
  `TileMatrixSet` key for the two values described above.

* Support for multi-level datasets aka ND image pyramids has been 
  further improved (#655):
  - Introduced new parameter `agg_methods` for writing multi-level datasets 
    with the "file", "s3", and "memory" data stores. 
    The value of `agg_methods` is either a string `"first"`,
    `"min"`, `"max"`, `"mean"`, `"median"` or a dictionary that maps
    a variable name to an aggregation method. Variable names can be patterns
    that may contain wildcard characters '*' and '?'. The special aggregation
    method `"auto"` can be used to select `"first"` for integer variables 
    and `"mean"` for floating point variables. 
  - The `xcube level` CLI tool now has a new option `--agg-methods` (or `-A`)
    for the same purpose.

* The xcube package now consistently makes use of logging.
  We distinguish general logging and specific xcube logging.
  General logging refers to the log messages emitted by any Python module 
  while xcube logging only refers to log messages emitted by xcube modules.

  * The output of general logging from xcube CLI tools can now be 
    configured with two new CLI options: 
    
    - `--loglevel LEVEL`: Can be one of `CRITICAL`, `ERROR`,
      `WARNING`, `INFO`, `DETAIL`, `DEBUG`, `TRACE`, or `OFF` (the default).
    - `--logfile PATH`: Effective only if log level is not `OFF`.
      If given, log messages will be written into the file
      given by PATH. If omitted, log messages will be redirected 
      to standard error (`sys.stderr`).

    The output of general logging from xcube CLI is disabled by default.
    If enabled, the log message format includes the level, date-time,
    logger name, and message.

  * All xcube modules use the logger named `xcube` 
    (i.e. `LOG = logging.getLogger("xcube")`) to emit 
    messages regarding progress, debugging, errors. Packages that extend
    the xcube package should use a dot suffix for their logger names, e.g.
    `xcube.cci` for the xcube plugin package `xcube-cci`.
  
  * All xcube CLI tools will output log messages, if any, 
    on standard error (`sys.stderr`). 
    Only the actual result, if any, 
    is written to standard out (`sys.stdout`).

  * Some xcube CLI tools have a `--quiet`/`-q` option to disable output
    of log messages on the console and a `--verbose`/`-v` option to enable 
    it and control the log level. For this purpose the option `-v` 
    can be given multiple times and even be combined: `-v` = `INFO`, 
    `-vv` = `DETAIL`, `-vvv` = `DEBUG`, `-vvvv` = `TRACE`.
    The `quiet` and `verbose` settings only affect the logger named `xcube`
    and its children. 
    If enabled, a simple message format will be used, unless the general 
    logging is redirected to stdout.

### Fixes

* Fixed a problem where the `DataStores` configuration of `xcube serve` 
  did not recognize multi-level datasets. (#653)

* Opening of multi-level datasets with filesystem data stores now 
  recognizes the `cache_size` open parameter.

* It is possible again to build and run docker containers from the docker file 
  in the Github Repository. (#651)
  For more information, see 
  https://xcube.readthedocs.io/en/latest/installation.html#docker 

### Other changes

* The `xcube tile` CLI tool has been deprecated. A new tool is planned that can work
  concurrently on dask clusters and also supports common tile grids such as
  global geographic and web mercator.

* The `xcube.util.tiledimage` module has been deprecated and is no longer 
  used in xcube. It has no replacement.

* The `xcube.util.tilegrid` module has been deprecated and is no longer 
  used in xcube. 
  A new implementation is provided by `xcube.core.tilingscheme` 
  which is used instead. 

* All existing functions of the `xcube.core.tile` module have been 
  deprecated and are no longer used in xcube. A newly exported function
  is `xcube.core.tile.compute_rgba_tile()` which is used in place of
  other tile generating functions.
  

## Changes in 0.10.2

### Enhancements

* Added new module `xcube.core.subsampling` for function
  `subsample_dataset(dataset, step)` that is now used by default 
  to generate the datasets level of multi-level datasets.

* Added new setting `Authentication.IsRequired` to the `xcube serve` 
  configuration. If set to `true`, xcube Server will reject unauthorized 
  dataset requests by returning HTTP code 401.
  
* For authorized clients, the xcube Web API provided by `xcube serve`
  now allows granted scopes to contain wildcard characters `*`, `**`,
  and `?`. This is useful to give access to groups of datasets, e.g.
  the scope `read:dataset:*/S2-*.zarr` permits access to any Zarr 
  dataset in a subdirectory of the configured data stores and 
  whose name starts with "S2-". (#632)

* `xcube serve` used to shut down with an error message 
  if it encountered datasets it could not open. New behaviour 
  is to emit a warning and ignore such datasets. (#630)

* Introduced helper function `add_spatial_ref()`
  of package `xcube.core.gridmapping.cfconv` that allows 
  adding a spatial coordinate reference system to an existing  
  Zarr dataset. (#629)

* Support for multi-level datasets has been improved:
  - Introduced new parameters for writing multi-level datasets with the 
    "file", "s3", and "memory" data stores (#617). They are 
    + `base_dataset_id`: If given, the base dataset will be linked only 
      with the value of `base_dataset_id`, instead of being copied as-is.
      This can save large amounts of storage space. 
    + `tile_size`: If given, it forces the spatial dimensions to be 
       chunked accordingly. `tile_size` can be a positive integer 
       or a pair of positive integers.
    + `num_levels`: If given, restricts the number of resolution levels 
       to the given value. Must be a positive integer to be effective.
  - Added a new example notebook 
    [5_multi_level_datasets.ipynb](https://github.com/dcs4cop/xcube/blob/main/examples/notebooks/datastores/5_multi_level_datasets.ipynb) 
    that demonstrates writing and opening multi-level datasets with the 
    xcube filesystem data stores.
  - Specified [xcube Multi-Resolution Datasets](https://github.com/dcs4cop/xcube/blob/main/docs/source/mldatasets.md)
    definition and format.

* `xcube gen2` returns more expressive error messages.
  
### Fixes

* Fixed problem where the dataset levels of multi-level datasets were 
  written without spatial coordinate reference system. In fact, 
  only spatial variables were written. (#646)

* Fixed problem where xcube Server instances that required 
  user authentication published datasets and variables for 
  unauthorised users.

* Fixed `FsDataAccessor.write_data()` implementations, 
  which now always return the passed in `data_id`. (#623)

* Fixes an issue where some datasets seemed to be shifted in the 
  y-(latitude-) direction and were misplaced on maps whose tiles 
  are served by `xcube serve`. Images with ascending y-values are 
  now tiled correctly. (#626)

### Other

* The `xcube level` CLI tool has been rewritten from scratch to make use 
  of xcube filesystem data stores. (#617)

* Deprecated numerous classes and functions around multi-level datasets.
  The non-deprecated functions and classes of `xcube.core.mldataset` should 
  be used instead along with the xcube filesystem data stores for 
  multi-level dataset i/o. (#516)
  - Deprecated all functions of the `xcube.core.level` module
    + `compute_levels()`
    + `read_levels()`
    + `write_levels()`
  - Deprecated numerous classes and functions of the `xcube.core.mldataset`
    module
    + `FileStorageMultiLevelDataset`
    + `ObjectStorageMultiLevelDataset`
    + `open_ml_dataset()`
    + `open_ml_dataset_from_object_storage()`
    + `open_ml_dataset_from_local_fs()`
    + `write_levels()`

* Added packages `python-blosc` and `lz4` to the xcube Python environment 
  for better support of Dask `distributed` and the Dask service 
  [Coiled](https://coiled.io/).

* Replace the dependency on the `rfc3339-validator` PyPI package with a
  dependency on its recently created conda-forge package.

* Remove unneeded dependency on the no longer used `strict-rfc3339` package.

## Changes in 0.10.1

### Fixes

* Deprecated argument `xy_var_names` in function `GridMapping.from_dataset`,
  thereby preventing a NotImplementedError. (#551) 

### Other Changes

* For compatibility, now also `xcube.__version__` contains the xcube 
  version number.

## Changes in 0.10.0

### Incompatible Changes 

* The configuration `DataStores` for `xcube serve` changed in an
  incompatible way with xcube 0.9.x: The value of former `Identifier` 
  must now be assigned to `Path`, which is a mandatory parameter. 
  `Path` may contain wildcard characters \*\*, \*, ?. 
  `Identifier` is now optional, the default is 
  `"${store_id}~${data_id}"`. If given, it should only be used to 
  uniquely identify single datasets within a data store
  pointed to by `Path`. (#516) 

### Enhancements

* It is now possible to use environment variables in most  
  xcube configuration files. Unix bash syntax is used, i.e. 
  `${ENV_VAR_NAME}` or `$ENV_VAR_NAME`. (#580)
  
  Supported tools include
  - `xcube gen --config CONFIG` 
  - `xcube gen2 --stores STORES_CONFIG --service SERVICE_CONFIG` 
  - `xcube serve -c CONFIG` 

* Changed the `xcube gen` tool to extract metadata for pre-sorting inputs
  from other than NetCDF inputs, e.g. GeoTIFF.

* Optimized function `xcube.core.geom.rasterize_features()`.
  It is now twice as fast while its memory usage dropped to the half. (#593)
  
### Fixes

* `xcube serve` now also serves datasets that are located in 
  subdirectories of filesystem-based data stores such as
  "file", "s3", "memory". (#579)

* xcube serve now accepts datasets whose spatial 
  resolutions differ up to 1%. (#590)
  It also no longer rejects datasets with large dimension 
  sizes. (Formerly, an integer-overflow occurred in size 
  computation.) 

* `DatasetChunkCacheSize` is now optional in `xcube serve`
  configuration. (Formerly, when omitted, the server crashed.)
  
* Fixed bug that would cause that requesting data ids on some s3 stores would
  fail with a confusing ValueError.
  
* Fixed that only last dataset of a directory listing was published via 
  `xcube serve` when using the `DataStores` configuration with 
  filesystem-based datastores such as "s3" or "file". (#576)
  
### Other

* Pinned Python version to < 3.10 to avoid import errors caused by a 
  third-party library.

* Values `obs` and `local` for the `FileSystem` parameter in xcube 
  configuration files have been replaced by `s3` and `file`, but are kept 
  temporarily for the sake of backwards compatibility.

## Changes in 0.9.2

### Fixes

* A `xcube.core.store.fs.impl.FSDataStore` no longer raises exceptions when 
  root directories in data store configurations do not exist. Instead, they 
  are created when data is written.

## Changes in 0.9.1

### New features

* The `xcube.core.maskset.MaskSet` class no longer allocates static numpy 
  arrays for masks. Instead, it uses lazy dask arrays. (#556)

* Function `xcube.core.geom.mask_dataset_by_geometry` has a new parameter 
  `all_touched`: If `True`, all pixels intersected by geometry outlines will 
  be included in the mask. If `False`, only pixels whose center is within the 
  polygon or that are selected by Bresenham’s line algorithm will be included  
  in the mask. The default value is set to `False`. 

### Other

* Updated `Dockerfile`: Removed the usage of a no-longer-maintained base image.
  Ensured that the version tag 'latest' can be used with installation mode 
  'release' for xcube plugins.

* The `xcube` package now requires `xarray >= 0.19`, `zarr >= 2.8`, 
  `pandas >= 1.3`.

## Changes in 0.9.0

### New features

* The implementations of the default data stores `s3`, `directory`, 
  and `memory` have been replaced entirely by a new implementation
  that utilize the [fsspec](https://filesystem-spec.readthedocs.io/) 
  Python package. The preliminary filesystem-based data stores 
  are now `s3`, `file`, and `memory`. All share a common implementations 
  and tests. Others filesystem-based data stores can be added easily
  and will follow soon, for example `hdfs`. 
  All filesystem-based data stores now support xarray
  datasets (type `xarray.Dataset`) in Zarr and NetCDF format as 
  well as image pyramids (type`xcube.core.multilevel.MultiLevelDataset`) 
  using a Zarr-based multi-level format. (#446)

* Several changes became necessary on the xcube Generator
  package `xcube.core.gen2` and CLI `xcube gen2`. 
  They are mostly not backward compatible:
  - The only supported way to instantiate cube generators is the
    `CubeGenerator.new()` factory method. 
  - `CubeGenerator.generate_cube()` and `CubeGenerator.get_cube_info()`
    both now receive the request object that has formerly been passed 
    to the generator constructors.
  - The `CubeGenerator.generate_cube()` method now returns a 
    `CubeGeneratorResult` object rather than a simple string 
    (the written `data_id`).  
  - Empty cubes are no longer written, a warning status is 
    generated instead.
  - The xcube gen2 CLI `xcube gen2` has a new option `--output RESULT` 
    to write the result to a JSON file. If it is omitted, 
    the CLI will dump the result as JSON to stdout.

* Numerous breaking changes have been applied to this version
  in order to address generic resampling (#391), to support other
  CRS than WGS-84 (#112), and to move from the struct data cube 
  specification to a more relaxed cube convention (#488): 
  * The following components have been removed entirely 
    - module `xcube.core.imgeom` with class `ImageGeom` 
    - module `xcube.core.geocoding` with class `GeoCoding`
    - module `xcube.core.reproject` and all its functions
  * The following components have been added 
    - module `xcube.core.gridmapping` with new class `GridMapping`
      is a CF compliant replacement for classes `ImageGeom` and `GeoCoding`
  * The following components have changed in an incompatible way:
    - Function`xcube.core.rectify.rectify_dataset()` now uses 
      `source_gm: GridMapping` and `target_gm: GridMapping` instead of 
      `geo_coding: GeoCoding` and `output_geom: ImageGeom`. 
    - Function`xcube.core.gen.iproc.InputProcessor.process()` now uses 
      `source_gm: GridMapping` and `target_gm: GridMapping` instead of 
      `geo_coding: GeoCoding` and `output_geom: ImageGeom`. 
  * xcube no longer depends on GDAL (at least not directly).
    
* Added a new feature to xcube called "BYOA" - Bring your own Algorithm.
  It is a generic utility that allows for execution of user-supplied 
  Python code in both local and remote contexts. (#467)
  The new `xcube.core.byoa` package hosts the BYOA implementation and API.
  The entry point to the functionality is the `xcube.core.byoa.CodeConfig`
  class. It is currently utilized by the xcube Cube Generator that can now
  deal with an optional `code_config` request parameter. If given,
  the generated data cube will be post-processed by the configured user-code.
  The xcube Cube Generator with the BYOA feature is made available through the 
  1. Generator API `xcube.core.gen2.LocalCubeGenerator` and
    `xcube.core.gen2.service.RemoteCubeGenerator`;
  2. Generator CLI `xcube gen2`.
  
* A dataset's cube subset and its grid mapping can now be accessed through
  the `xcube` property of `xarray.Dataset` instances. This feature requires 
  importing the `xcube.core.xarray`package. Let `dataset` be an 
  instance of `xarray.Dataset`, then
  - `dataset.xcube.cube` is a `xarray.Dataset` that contains all cube 
     variables of `dataset`, namely the ones with dimensions 
     `("time", [...,], y_dim_name, x_dim_name)`, where `y_dim_name`, 
    `x_dim_name` are determined by the dataset's grid mapping.
     May be empty, if `dataset` has no cube variables.
  - `dataset.xcube.gm` is a `xcube.core.gridmapping.GridMapping` that 
     describes the CF-compliant grid mapping of `dataset`. 
     May be `None`, if `dataset` does not define a grid mapping.
  - `dataset.xcube.non_cube` is a `xarray.Dataset` that contains all
     variables of `dataset` that are not in `dataset.xcube.cube`.
     May be same as `dataset`, if `dataset.xcube.cube` is empty.
  
* Added a new utility module `xcube.util.temp` that allows for creating 
  temporary files and directories that will be deleted when the current 
  process ends.
* Added function `xcube.util.versions.get_xcube_versions()`  
  that outputs the versions of packages relevant for xcube.
  Also added a new CLI `xcube versions` that outputs the result of the  
  new function in JSON or YAML. (#522)

### Other

* The xcube cube generator (API `xcube.core.gen2`, CLI `xcube gen2`) 
  will now write consolidated Zarrs by default. (#500)
* xcube now issues a warning, if a data cube is opened from object 
  storage, and credentials have neither been passed nor can be found, 
  and the object storage has been opened with the default `anon=False`. (#412)
* xcube no longer internally caches directory listings, which prevents 
  the situation where a data cube that has recently been written into object 
  storage cannot be found. 
* Removed example notebooks that used hard-coded local file paths. (#400)
* Added a GitHub action that will run xcube unit tests, and build and 
  push Docker images. The version tag of the image is either `latest` when 
  the main branch changed, or the same as the release tag. 
* Removed warning `module 'xcube_xyz' looks like an xcube-plugin but 
  lacks a callable named 'init_plugin`.
* Fixed an issue where `xcube serve` provided wrong layer source options for 
  [OpenLayers XYZ](https://openlayers.org/en/latest/apidoc/module-ol_source_XYZ-XYZ.html) 
  when latitude coordinates where increasing with the coordinate index. (#251)
* Function `xcube.core.normalize.adjust_spatial_attrs()` no longer removes
  existing global attributes of the form `geospatial_vertical_<property>`.
* Numerous classes and functions became obsolete in the xcube 0.9 
  code base and have been removed, also because we believe there is 
  quite rare outside use, if at all. 
  
  Removed from `xcube.util.tiledimage`:
  * class `DownsamplingImage`
  * class `PilDownsamplingImage`
  * class `NdarrayDownsamplingImage`
  * class `FastNdarrayDownsamplingImage`
  * class `ImagePyramid`
  * function `create_pil_downsampling_image()`
  * function `create_ndarray_downsampling_image()`
  * function `downsample_ndarray()`
  * functions `aggregate_ndarray_xxx()`
  
  Removed from `xcube.util.tilegrid`:
  * functions `pow2_2d_subdivision()`
  * functions `pow2_1d_subdivision()`
  
## Changes in 0.8.2

* Fixed the issue that xcube gen2 would not print tracebacks to stderr 
  when raising errors of type `CubeGeneratorError` (#448).
* Enhanced `xcube.core.normalize.normalize_dataset()` function to also 
  normalize datasets with latitudes given as 
  `latitude_centers` and to invert decreasing latitude coordinate values.
* Introduced `xcube.core.normalize.cubify_dataset()` function to normalize 
  a dataset and finally assert the result complies to the 
  [xcube dataset conventions](https://github.com/dcs4cop/xcube/blob/main/docs/source/cubespec.md).
* Fixed that data stores `directory` and `s3` were not able to handle data 
  identifiers that they had assigned themselves during `write_data()`.  
  (#450)
* The `xcube prune` tool is no longer restricted to data cube datasets 
  and should now be able to deal with datasets that comprise very many 
  chunks. (#469)
* The `xcube.core.extract.get_cube_values_for_points()` function has been 
  enhanced to also accept lists or tuples in the item values of 
  the `points` arguments. (#431)   
* Fixed exception raised in `xcube extract` CLI tool when called with the 
  `--ref` option. This issue occurred with `xarray 0.18.2+`.

## Changes in 0.8.1

* Improved support of datasets with time given as `cftime.DatetimeGregorian` 
  or `cftime.DatetimeJulian`.
* Fixed out-of-memory error raised if spatial subsets were created from 
  cubes with large spatial dimensions. (#442)
* Fixed example Notebook `compute_dask_array` and renamed it 
  into `compute_array_from_func`. (#385)
* Fixed a problem with the S3 data store that occurred if the store was 
  configured without `bucket_name` and the (Zarr) data was opened 
  with `consolidated=True`.

* The functions `xcube.core.compute.compute_cube()` 
  and `xcube.core.compute.compute_dataset()`
  can now alter the shape of input datasets. (#289)  

## Changes in 0.8.0

* Harmonized retrieval of spatial and temporal bounds of a dataset: 
  To determine spatial bounds, use `xcube.core.geom.get_dataset_bounds()`, 
  to determine temporal bounds, use `xcube.core.timecoord.get_time_range_from_data()`. 
  Both methods will attempt to get the values from associated bounds arrays first. 
* Fixed broken JSON object serialisation of objects returned by 
  `DataStore.describe_object()`. (#432)
* Changed behaviour and signature of `xcube.core.store.DataStore.get_dataset_ids()`.
  The keyword argument `include_titles: str = True` has been replaced by 
  `include_attrs: Sequence[str] = None` and the return value changes accordingly:
  - If `include_attrs` is None (the default), the method returns an iterator
    of dataset identifiers *data_id* of type `str`.
  - If `include_attrs` is a sequence of attribute names, the method returns
    an iterator of tuples (*data_id*, *attrs*) of type `Tuple[str, Dict]`.
  Hence `include_attrs`  can be used to obtain a minimum set of dataset 
  metadata attributes for each returned *data_id*.
  However, `include_attrs` is not yet implemented so far in the "s3", 
  "memory", and "directory" data stores. (#420)
* Directory and S3 Data Store consider format of data denoted by *data id* when 
  using `get_opener_ids()`.
* S3 Data Store will only recognise a `consolidated = True` parameter setting,
  if the file `{bucket}/{data_id}/.zmetadata` exists. 
* `xcube gen2` will now ensure that temporal subsets can be created. (#430)
* Enhance `xcube serve` for use in containers: (#437)
  * In addition to option `--config` or `-c`, dataset configurations can now 
    be passed via environment variable `XCUBE_SERVE_CONFIG_FILE`.
  * Added new option `--base-dir` or `-b` to pass the base directory to
    resolve relative paths in dataset configurations. In addition, the value
    can be passed via environment variable `XCUBE_SERVE_BASE_DIR`.

## Changes in 0.7.2

* `xcube gen2` now allows for specifying the final data cube's chunk
  sizes. The new `cube_config` parameter is named `chunks`, is optional
  and if given, must be a dictionary that maps a dimension name to a 
  chunk size or to `None` (= no chunking). The chunk sizes only apply 
  to data variables. Coordinate variables will not be affected, e.g. 
  "time", "lat", "lon" will not be chunked. (#426)

* `xcube gen2` now creates subsets from datasets returned by data stores that
  do not recognize cube subset parameters `variable_names`, `bbox`, and
  `time_range`. (#423)

* Fixed a problem where S3 data store returned outdated bucket items. (#422)

## Changes in 0.7.1

* Dataset normalisation no longer includes reordering increasing
  latitude coordinates, as this creates datasets that are no longer writable 
  to Zarr. (#347)
* Updated package requirements
  - Added `s3fs`  requirement that has been removed by accident.
  - Added missing requirements `requests` and `urllib3`.

## Changes in 0.7.0

* Introduced abstract base class `xcube.util.jsonschema.JsonObject` which 
  is now the super class of many classes that have JSON object representations.
  In Jupyter notebooks, instances of such classes are automatically rendered 
  as JSON trees.
* `xcube gen2` CLI tool can now have multiple `-v` options, e.g. `-vvv`
  will now output detailed requests and responses.  
* Added new Jupyter notebooks in `examples/notebooks/gen2` 
  for the _data cube generators_ in the package `xcube.core.gen2`.
* Fixed a problem in `JsonArraySchema` that occurred if a valid 
  instance was `None`. A TypeError `TypeError: 'NoneType' object is not iterable` was 
  raised in this case.
* The S3 data store  `xcube.core.store.stores.s3.S3DataStore` now implements the `describe_data()` method. 
  It therefore can also be used as a data store from which data is queried and read.  
* The `xcube gen2` data cube generator tool has been hidden from
  the set of "official" xcube tools. It is considered as an internal tool 
  that is subject to change at any time until its interface has stabilized.
  Please refer to `xcube gen2 --help` for more information.
* Added `coords` property to `DatasetDescriptor` class. 
  The `data_vars` property of the `DatasetDescriptor` class is now a dictionary. 
* Added `chunks` property to `VariableDescriptor` class. 
* Removed function `reproject_crs_to_wgs84()` and tests (#375) because  
  - it seemed to be no longer be working with GDAL 3.1+; 
  - there was no direct use in xcube itself;
  - xcube plans to get rid of GDAL dependencies.
* CLI tool `xcube gen2` may now also ingest non-cube datasets.
* Fixed unit tests broken by accident. (#396)
* Added new context manager `xcube.util.observe_dask_progress()` that can be used
  to observe tasks that known to be dominated by Dask computations: 
  ```python
  with observe_dask_progress('Writing dataset', 100):
      dataset.to_zarr(store)  
  ```
* The xcube normalisation process, which ensures that a dataset meets the requirements 
  of a cube, internally requested a lot of data, causing the process to be slow and
  expensive in terms of memory consumption. This problem was resolved by avoiding to
  read in these large amounts of data. (#392)

## Changes in 0.6.1

* Updated developer guide (#382)

Changes relating to maintenance of xcube's Python environment requirements in `envrionment.yml`:

* Removed explicit `blas` dependency (which required MKL as of `blas =*.*=mkl`) 
  for better interoperability with existing environments.  
* Removed restrictions of `fsspec <=0.6.2` which was required due to 
  [Zarr #650](https://github.com/zarr-developers/zarr-python/pull/650). As #650 has been fixed, 
  `zarr=2.6.1` has been added as new requirement. (#360)

## Changes in 0.6.0

### Enhancements 

* Added four new Jupyter Notebooks about xcube's new Data Store Framework in 
  `examples/notebooks/datastores`.

* CLI tool `xcube io dump` now has new `--config` and `--type` options. (#370)

* New function `xcube.core.store.get_data_store()` and new class `xcube.core.store.DataStorePool` 
  allow for maintaining a set of pre-configured data store instances. This will be used
  in future xcube tools that utilise multiple data stores, e.g. "xcube gen", "xcube serve". (#364)

* Replaced the concept of `type_id` used by several `xcube.core.store.DataStore` methods 
  by a more flexible `type_specifier`. Documentation is provided in `docs/source/storeconv.md`. 
  
  The `DataStore` interface changed as follows:
  - class method `get_type_id()` replaced by `get_type_specifiers()` replaces `get_type_id()`;
  - new instance method `get_type_specifiers_for_data()`;
  - replaced keyword-argument in `get_data_ids()`;
  - replaced keyword-argument in `has_data()`;
  - replaced keyword-argument in `describe_data()`;
  - replaced keyword-argument in `get_search_params_schema()`;
  - replaced keyword-argument in `search_data()`;
  - replaced keyword-argument in `get_data_opener_ids()`.
  
  The `WritableDataStore` interface changed as follows:
  - replaced keyword-argument in `get_data_writer_ids()`.

* The JSON Schema classes in `xcube.util.jsonschema` have been extended:
  - `date` and `date-time` formats are now validated along with the rest of the schema
  - the `JsonDateSchema` and `JsonDatetimeSchema` subclasses of `JsonStringSchema` have been introduced, 
    including a non-standard extension to specify date and time limits

* Extended `xcube.core.store.DataStore` docstring to include a basic convention for store 
  open parameters. (#330)

* Added documentation for the use of the open parameters passed to 
  `xcube.core.store.DataOpener.open_data()`.

### Fixes

* `xcube serve` no longer crashes, if configuration is lacking a `Styles` entry.

* `xcube gen` can now interpret `start_date` and `stop_date` from NetCDF dataset attributes. 
  This is relevant for using `xcube gen` for Sentinel-2 Level 2 data products generated and 
  provided by Brockmann Consult. (#352)


* Fixed both `xcube.core.dsio.open_cube()` and `open_dataset()` which failed with message 
  `"ValueError: group not found at path ''"` if called with a bucket URL but no credentials given
  in case the bucket is not publicly readable. (#337)
  The fix for that issue now requires an additional `s3_kwargs` parameter when accessing datasets 
  in _public_ buckets:
  ```python
  from xcube.core.dsio import open_cube 
    
  public_url = "https://s3.eu-central-1.amazonaws.com/xcube-examples/OLCI-SNS-RAW-CUBE-2.zarr"
  public_cube = open_cube(public_url, s3_kwargs=dict(anon=True))
  ```  
* xcube now requires `s3fs >= 0.5` which implies using faster async I/O when accessing object storage.
* xcube now requires `gdal >= 3.0`. (#348)
* xcube now only requires `matplotlib-base` package rather than `matplotlib`. (#361)

### Other

* Restricted `s3fs` version in envrionment.yml in order to use a version which can handle pruned xcube datasets.
  This restriction will be removed once changes in zarr PR https://github.com/zarr-developers/zarr-python/pull/650 
  are merged and released. (#360)
* Added a note in the `xcube chunk` CLI help, saying that there is a possibly more efficient way 
  to (re-)chunk datasets through the dedicated tool "rechunker", see https://rechunker.readthedocs.io
  (thanks to Ryan Abernathey for the hint). (#335)
* For `xcube serve` dataset configurations where `FileSystem: obs`, users must now also 
  specify `Anonymous: True` for datasets in public object storage buckets. For example:
  ```yaml
  - Identifier: "OLCI-SNS-RAW-CUBE-2"
    FileSystem: "obs"
    Endpoint: "https://s3.eu-central-1.amazonaws.com"
    Path: "xcube-examples/OLCI-SNS-RAW-CUBE-2.zarr"
    Anyonymous: true
    ...
  - ...
  ```  
* In `environment.yml`, removed unnecessary explicit dependencies on `proj4` 
  and `pyproj` and restricted `gdal` version to >=3.0,<3.1. 

## Changes in 0.5.1

* `normalize_dataset` now ensures that latitudes are decreasing.

## Changes in 0.5.0

### New 

* `xcube gen2 CONFIG` will generate a cube from a data input store and a user given cube configuration.
   It will write the resulting cube in a user defined output store.
    - Input Stores: CCIODP, CDS, SentinelHub
    - Output stores: memory, directory, S3

* `xcube serve CUBE` will now use the last path component of `CUBE` as dataset title.

* `xcube serve` can now be run with AWS credentials (#296). 
  - In the form `xcube serve --config CONFIG`, a `Datasets` entry in `CONFIG`
    may now contain the two new keys `AccessKeyId: ...` and `SecretAccessKey: ...` 
    given that `FileSystem: obs`.
  - In the form `xcube serve --aws-prof PROFILE CUBE`
    the cube stored in bucket with URL `CUBE` will be accessed using the
    credentials found in section `[PROFILE]` of your `~/.aws/credentials` file.
  - In the form `xcube serve --aws-env CUBE`
    the cube stored in bucket with URL `CUBE` will be accessed using the
    credentials found in environment variables `AWS_ACCESS_KEY_ID` and
    `AWS_SECRET_ACCESS_KEY`.


* xcube has been extended by a new *Data Store Framework* (#307).
  It is provided by the `xcube.core.store` package.
  It's usage is currently documented only in the form of Jupyter Notebook examples, 
  see `examples/store/*.ipynb`.
   
* During the development of the new *Data Store Framework*, some  
  utility packages have been added:
  * `xcube.util.jsonschema` - classes that represent JSON Schemas for types null, boolean,
     number, string, object, and array. Schema instances are used for JSON validation,
     and object marshalling.
  * `xcube.util.assertions` - numerous `assert_*` functions that are used for function 
     parameter validation. All functions raise `ValueError` in case an assertion is not met.
  * `xcube.util.ipython` - functions that can be called for better integration of objects with
     Jupyter Notebooks.

### Enhancements

* Added possibility to specify packing of variables within the configuration of
  `xcube gen` (#269). The user now may specify a different packing variables, 
  which might be useful for reducing the storage size of the datacubes.
  Currently it is only implemented for zarr format.
  This may be done by passing the parameters for packing as the following:  
   
   
  ```yaml  
  output_writer_params: 

    packing: 
      analysed_sst: 
        scale_factor: 0.07324442274239326
        add_offset: -300.0
        dtype: 'uint16'
        _FillValue: 0.65535
  ```

* Example configurations for `xcube gen2` were added.

### Fixes

* From 0.4.1: Fixed time-series performance drop (#299). 

* Fixed `xcube gen` CLI tool to correctly insert time slices into an 
  existing cube stored as Zarr (#317).  

* When creating an ImageGeom from a dataset, correct the height if it would
  otherwise give a maximum latitude >90°.

* Disable the display of warnings in the CLI by default, only showing them if
  a `--warnings` flag is given.

* Fixed a regression when running "xcube serve" with cube path as parameter (#314)

* From 0.4.3: Extended `xcube serve` by reverse URL prefix option. 

* From 0.4.1: Fixed time-series performance drop (#299). 


## Changes in 0.4.3

* Extended `xcube serve` by reverse URL prefix option `--revprefix REFPREFIX`.
  This can be used in cases where only URLs returned by the service need to be prefixed, 
  e.g. by a web server's proxy pass.

## Changes in 0.4.2 

* Fixed a problem during release process. No code changes.

## Changes in 0.4.1 

* Fixed time-series performance drop (#299). 

## Changes in 0.4.0

### New

* Added new `/timeseries/{dataset}/{variable}` POST operation to xcube web API.
  It extracts time-series for a given GeoJSON object provided as body.
  It replaces all of the `/ts/{dataset}/{variable}/{geom-type}` operations.
  The latter are still maintained for compatibility with the "VITO viewer". 
  
* The xcube web API provided through `xcube serve` can now serve RGBA tiles using the 
  `dataset/{dataset}/rgb/tiles/{z}/{y}/{x}` operation. The red, green, blue 
  channels are computed from three configurable variables and normalisation ranges, 
  the alpha channel provides transparency for missing values. To specify a default
  RGB schema for a dataset, a colour mapping for the "pseudo-variable" named `rbg` 
  is provided in the configuration of `xcube serve`:
  ```yaml  
  Datasets:
    - Identifyer: my_dataset
      Style: my_style
      ...
    ...
  Styles:
    - Identifier: my_style
      ColorMappings:
        rgb:
          Red:
            Variable: rtoa_8
            ValueRange: [0., 0.25]
          Green:
            Variable: rtoa_6
            ValueRange: [0., 0.25]
          Blue:
            Variable: rtoa_4
            ValueRange: [0., 0.25]
        ...
  ```
  Note that this concept works nicely in conjunction with the new `Augmentation` feature (#272) used
  to compute new variables that could be input to the RGB generation. 
  
* Introduced new (ortho-)rectification algorithm allowing reprojection of 
  satellite images that come with (terrain-corrected) geo-locations for every pixel.

  - new CLI tool `xcube rectify`
  - new API function `xcube.core.rectify.rectify_dataset()`

* Utilizing the new rectification in `xcube gen` tool. It is now the default 
  reprojection method in `xcube.core.gen.iproc.XYInputProcessor` and
  `xcube.core.gen.iproc.DefaultInputProcessor`, if ground control points are not 
  specified, i.e. the input processor is configured with `xy_gcp_step=None`. (#206)
* Tile sizes for rectification in `xcube gen` are now derived from `output_writer_params` if given in configuration and 
  if it contains a `chunksizes` parameter for 'lat' or 'lon'. This will force the generation of a chunked xcube dataset 
  and will utilize Dask arrays for out-of-core computations. This is very useful for large data cubes whose time slices 
  would otherwise not fit into memory.
* Introduced new function `xcube.core.select.select_spatial_subset()`.

* Renamed function `xcube.core.select.select_vars()` into `xcube.core.select.select_variables_subset()`.
  
* Now supporting xarray and numpy functions in expressions used by the
  `xcube.core.evaluate.evaluate_dataset()` function and in the configuration of the 
  `xcube gen` tool. You can now use `xr` and `np` contexts in expressions, e.g. 
  `xr.where(CHL >= 0.0, CHL)`. (#257)

* The performance of the `xcube gen` tool for the case that expressions or 
  expression parts are reused across multiple variables can now be improved. 
  Such as expressions can now be assigned to intermediate variables and loaded 
  into memory, so they are not recomputed again.
  For example, let the expression `quality_flags.cloudy and CHL > 25.0` occur often
  in the configuration, then this is how recomputation can be avoided:
  ```
    processed_variables:
      no_cloud_risk:
        expression: not (quality_flags.cloudy and CHL_raw > 25.0)
        load: True
      CHL:
        expression: CHL_raw
        valid_pixel_expression: no_cloud_risk
      ...        
  ```      
* Added ability to write xcube datasets in Zarr format into object storage bucket using the xcube python api
  `xcube.core.dsio.write_cube()`. (#224) The user needs to pass provide user credentials via 
  ```
  client_kwargs = {'provider_access_key_id': 'user_id', 'provider_secret_access_key': 'user_secret'}
  ```
  and 
  write to existing bucket by executing 
  
  ```
  write_cube(ds1, 'https://s3.amazonaws.com/upload_bucket/cube-1-250-250.zarr', 'zarr',
                       client_kwargs=client_kwargs)
  ```
* Added new CLI tool `xcube tile` which is used to generate a tiled RGB image 
  pyramid from any xcube dataset. The format and file organisation of the generated 
  tile sets conforms to the [TMS 1.0 Specification](https://wiki.osgeo.org/wiki/Tile_Map_Service_Specification) 
  (#209).

* The configuration of `xcube serve` has been enhanced to support
  augmentation of data cubes by new variables computed on-the-fly (#272).
  You can now add a section `Augmentation` into a dataset descriptor, e.g.:
  
  ```yaml 
    Datasets:
      - Identifier: abc
        ...
        Augmentation:
          Path: compute_new_vars.py
          Function: compute_variables
          InputParameters:
            ...
      - ...
  ```
  
  where `compute_variables` is a function that receives the parent xcube dataset
  and is expected to return a new dataset with new variables. 
  
* The `xcube serve` tool now provides basic access control via OAuth2 bearer tokens (#263).
  To configure a service instance with access control, add the following to the 
  `xcube serve` configuration file:
  
  ```
    Authentication:
      Domain: "<your oauth2 domain>"
      Audience: "<your audience or API identifier>"
  ```
  
  Individual datasets can now be protected using the new `AccessControl` entry
  by configuring the `RequiredScopes` entry whose value is a list
  of required scopes, e.g. "read:datasets":
  
  ```
    Datasets:
      ...
      - Identifier: <some dataset id>
        ...
        AccessControl:
          RequiredScopes:
            - "read:datasets"
  ```
  
  If you want a dataset to disappear for authorized requests, set the 
  `IsSubstitute` flag:
  
  ```
    Datasets:
      ...
      - Identifier: <some dataset id>
        ...
        AccessControl:
          IsSubstitute: true
  ```

### Enhancements

* The `xcube serve` tool now also allows for per-dataset configuration
  of *chunk caches* for datasets read from remote object storage locations. 
  Chunk caching avoids recurring fetching of remote data chunks for same
  region of interest.
  It can be configured as default for all remote datasets at top-level of 
  the configuration file:
  ```
  DatasetChunkCacheSize: 100M
  ```
  or in individual dataset definitions:
  ```
  Datasets: 
     - Identifier: ...
       ChunkCacheSize: 2G
       ...
  ```
* Retrieval of time series in Python API function `xcube.core.timeseries.get_time_series()` 
  has been optimized and is now much faster for point geometries. 
  This enhances time-series performance of `xcube serve`. 
  * The log-output of `xcube serve` now contains some more details time-series request 
    so performance bottlenecks can be identified more easily from `xcube-serve.log`, 
    if the server is started together with the flag `--traceperf`.
* CLI command `xcube resample` has been enhanced by a new value for the 
  frequency option `--frequency all`
  With this value it will be possible to create mean, max , std, ... of the whole dataset,
  in other words, create an overview of a cube. 
  By [Alberto S. Rabaneda](https://github.com/rabaneda).
 
* The `xcube serve` tool now also serves dataset attribution information which will be 
  displayed in the xcube-viewer's map. To add attribution information, use the `DatasetAttribution` 
  in to your `xcube serve` configuration. It can be used on top-level (for all dataset), 
  or on individual datasets. Its value may be a single text entry or a list of texts:
  For example: 
  ```yaml
  DatasetAttribution: 
    - "© by Brockmann Consult GmbH 2020, contains modified Copernicus Data 2019, processed by ESA."
    - "Funded by EU H2020 DCS4COP project."
  ```
* The `xcube gen` tool now always produces consolidated xcube datasets when the output format is zarr. 
  Furthermore when appending to an existing zarr xcube dataset, the output now will be consolidated as well. 
  In addition, `xcube gen` can now append input time slices to existing optimized (consolidated) zarr xcube datasets.
* The `unchunk_coords` keyword argument of Python API function 
  `xcube.core.optimize.optimize_dataset()` can now be a name, or list of names  
  of the coordinate variable(s) to be consolidated. If boolean ``True`` is used
  all variables will be consolidated.
* The `xcube serve` API operations `datasets/` and `datasets/{ds_id}` now also
  return the metadata attributes of a given dataset and it variables in a property
  named `attrs`. For variables we added a new metadata property `htmlRepr` that is
  a string returned by a variable's `var.data._repr_html_()` method, if any.
* Renamed default log file for `xcube serve` command to `xcube-serve.log`.
* `xcube gen` now immediately flushes logging output to standard out
  
## Changes in 0.3.1 

### Fixes

* Removing false user warning about custom SNAP colormaps when starting 
  `xcube serve` command.
  
## Changes in 0.3.0

### New

* Added new parameter in `xcube gen` called `--no_sort`. Using `--no_sort`, 
  the input file list wont be sorted before creating the xcube dataset. 
  If `--no_sort` parameter is passed, order the input list will be kept. 
  The parameter `--sort` is deprecated and the input files will be sorted 
  by default. 
* xcube now discovers plugin modules by module naming convention
  and by Setuptools entry points. See new chapter 
  [Plugins](https://xcube.readthedocs.io/en/latest/plugins.html) 
  in xcube's documentation for details. (#211)  
* Added new `xcube compute` CLI command and `xcube.core.compute.compute_cube()` API 
  function that can be used to generate an output cube computed from a Python
  function that is applied to one or more input cubes. Replaces the formerly 
  hidden `xcube apply` command. (#167) 
* Added new function `xcube.core.geom.rasterize_features()` 
  to rasterize vector-data features into a dataset. (#222)
* Extended CLI command `xcube verify` and API function `xcube.core.verify.verify_cube` to check whether spatial
  coordinate variables and their associated bounds variables are equidistant. (#231)
* Made xarray version 0.14.1 minimum requirement due to deprecation of xarray's `Dataset.drop`
  method and replaced it with `drop_sel` and `drop_vars` accordingly. 


### Enhancements

* CLI commands execute much faster now when invoked with the `--help` and `--info` options.
* Added `serverPID` property to response of web API info handler. 
* Functions and classes exported by following modules no longer require data cubes to use
  the `lon` and `lat` coordinate variables, i.e. using WGS84 CRS coordinates. Instead, the 
  coordinates' CRS may be a projected coordinate system and coordinate variables may be called
  `x` and `y` (#112):
  - `xcube.core.new`
  - `xcube.core.geom`
  - `xcube.core.schema`
  - `xcube.core.verify`
* Sometimes the cell bounds coordinate variables of a given coordinate variables are not in a proper, 
  [CF compliant](http://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/cf-conventions.html#cell-boundaries) 
  order, e.g. for decreasing latitudes `lat` the respective bounds coordinate
  `lat_bnds` is decreasing for `lat_bnds[:, 0]` and `lat_bnds[:, 1]`, but `lat_bnds[i, 0] < lat_bnds[i, 1]`
  for all `i`. xcube is now more tolerant w.r.t. to such wrong ordering of cell boundaries and will 
  compute the correct spatial extent. (#233)
* For `xcube serve`, any undefined color bar name will default to `"viridis"`. (#238)
    
 
### Fixes

* `xcube resample` now correctly re-chunks its output. By default, chunking of the 
  `time` dimension is set to one. (#212)

### Incompatible changes

The following changes introduce incompatibilities with former xcube 0.2.x 
versions. 

* The function specified by `xcube_plugins` entry points now receives an single argument of 
  type `xcube.api.ExtensionRegistry`. Plugins are asked to add their extensions
  to this registry. As an example, have a look at the default `xcube_plugins` entry points 
  in `./setup.py`.   
 
* `xcube.api.compute_dataset()` function has been renamed to 
  `xcube.api.evaluate_dataset()`. This has been done in order avoid confusion
  with new API function `xcube.api.compute_cube()`.
  
* xcube's package structure has been drastically changed: 
  - all of xcube's `__init__.py` files are now empty and no longer 
    have side effects such as sub-module aggregations. 
    Therefore, components need to be imported from individual modules.
  - renamed `xcube.api` into `xcube.core`
  - moved several modules from `xcube.util` into `xcube.core`
  - the new `xcube.constants` module contains package level constants
  - the new `xcube.plugin` module now registers all standard extensions
  - moved contents of module `xcube.api.readwrite` into `xcube.core.dsio`.
  - removed functions `read_cube` and `read_dataset` as `open_cube` and `open_dataset` are sufficient
  - all internal module imports are now absolute, rather than relative  

## Changes in 0.2.1

### Enhancements

- Added new CLI tool `xcube edit` and API function `xcube.api.edit_metadata`
  which allows editing the metadata of an existing xcube dataset. (#170)
- `xcube serve` now recognises xcube datasets with
  metadata consolidated by the `xcube opmimize` command. (#141)

### Fixes
- `xcube gen` now parses time stamps correcly from input data. (#207)
- Dataset multi-resolution pyramids (`*.levels` directories) can be stored in cloud object storage
  and are now usable with `xcube serve` (#179)
- `xcube optimize` now consolidates metadata only after consolidating
  coordinate variables. (#194)
- Removed broken links from `./README.md` (#197)
- Removed obsolete entry points from `./setup.py`.

## Changes in 0.2.0

### New

* Added first version of the [xcube documentation](https://xcube.readthedocs.io/) generated from
  `./docs` folder.

### Enhancements

* Reorganisation of the Documentation and Examples Section (partly addressing #106)
* Loosened python conda environment to satisfy conda-forge requirements
* xcube is now available as a conda package on the conda-forge channel. To install
  latest xcube package, you can now type: `conda install -c conda-forge xcube`
* Changed the unittesting code to minimize warnings reported by 3rd-party packages
* Making CLI parameters consistent and removing or changing parameter abbreviations
  in case they were used twice for different params. (partly addressing #91)
  For every CLI command which is generating an output a path must be provided by the
  option `-o`, `--output`. If not provided by the user, a default output_path is generated.
  The following CLI parameter have changed and their abbreviation is not enabled anymore : 

    - `xcube gen -v` is now only `xcube gen --vars` or `xcube gen --variables` 
    - `xcube gen -p` is now  `xcube gen -P` 
    - `xcube gen -i` is now  `xcube gen -I` 
    - `xcube gen -r` is now  `xcube gen -R`
    - `xcube gen -s` is now  `xcube gen -S` 
    - `xcube chunk -c`  is now  `xcube chunk -C`
    - `xcube level -l` is now `xcube level -L`
    - `xcube dump -v` is now `xcube dump --variable` or `xcube dump --var`
    - `xcube dump -e` is now `xcube dump -E` 
    - `xcube vars2dim -v` is now `xcube vars2dim --variable` or `xcube vars2dim --var`
    - `xcube vars2dim --var_name` is now `xcube vars2dim --variable` or `xcube vars2dim --var`
    - `xcube vars2dim -d` is now `xcube vars2dim -D` 
    - `xcube grid res -d` is now `xcube grid res -D`
    - `xcube grid res -c` is now `xcube grid res --cov` or `xcube grid res --coverage` 
    - `xcube grid res -n` is now `xcube grid res -N` or `xcube grid res --num_results` 
    - `xcube serve -p` is now `xcube serve -P` 
    - `xcube serve -a` is now `xcube serve -A` 
    
* Added option `inclStDev` and `inclCount` query parameters to `ts/{dataset}/{variable}/geometry` and derivates.
  If used with `inclStDev=1`, Xcube Viewer will show error bars for each time series point.
* `xcube.api.new_cube` function now accepts callables as values for variables.
  This allows to compute variable values depending on the (t, y, x) position
  in the cube. Useful for testing.
* `xcube.api` now exports the `MaskSet` class which is useful for decoding flag values encoding following the
  [CF conventions](http://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/cf-conventions.html#flags).
* Added new CLI tool `xcube optimize` and API function `xcube.api.optimize_dataset` 
  optimizes data cubes for cloud object storage deployment. (#141)
* Added two new spatial dataset operations to Python API `xcube.api` (#148):
  * `mask_dataset_by_geometry(dataset, geometry)` clip and mask a dataset by geometry
  * `clip_dataset_by_geometry(dataset, geometry)` just clip a dataset by geometry 
* Changed the dev version tag from 0.2.0.dev3 to 0.2.0.dev
* The behavior of web API `/datasets?details=1` has changed.
  The call no longer includes associated vector data as GeoJSON. Instead new API
  has beed added to fetch new vector data on demand:
  `/datasets/{dataset}/places` and `/datasets/{dataset}/places/{place}` (#130)
* `xcube serve` accepts custom SNAP colormaps. The path to a SAP .cpd file can be passed via the server
   configuration file with the paramter [ColorFile] instead of [ColorBar]. (#84)
* `xcube serve` can now be configured to serve cubes that are associated 
   with another cube with same data but different chunking (#115). 
   E.g. using chunks such as `time=512,lat=1,lon=1` can drastically improve 
   time-series extractions. 
   Have a look at the demo config in `xube/webapi/res/demo/config.yml`.     
* `xcube serve` does now offer a AWS S3 compatible data access API (#97):
   - List bucket objects: `/s3bucket`, see AWS 
     docs [GET](https://docs.aws.amazon.com/AmazonS3/latest/API/v2-RESTBucketGET.html)
   - Get bucket object: `/s3bucket/{ds_id}/{path}`, 
     see AWS docs [HEAD](https://docs.aws.amazon.com/AmazonS3/latest/API/RESTObjectHEAD.html) 
     and [GET](https://docs.aws.amazon.com/AmazonS3/latest/API/RESTObjectGET.html)
* `xcube serve` now verifies that a configured cube is valid once it is opened. (#107)
* Added new CLI command `xcube verify` performing xcube dataset verification. (#19)
* Reworked `xcube extract` to be finally useful and effective for point data extraction. (#102) 
* `xcube server`can now filter datasets by point coordinate, e.g. `/datasets?point=12.5,52.8`. (#50) 
* `xcube server`can now limit time series to a maximum number of 
  valid (not NaN) values. To activate, pass optional query parameter `maxValids` to the various `/ts`
  functions. The special value `-1` will restrict the result to contain only valid values. (#113) 
* Reworked `xcube gen` to be more user-friendly and more consistent with other tools. 
  The changes are
  - Removed `--dir` and `--name` options and replaced it by single `--output` option, 
    whose default value is `out.zarr`. (#45)
  - The `--format` option no longer has a default value. If not given, 
    format is guessed from `--output` option.
  - Renamed following parameters in the configuration file:
    + `input_files` into `input_paths`, also because paths may point into object storage 
      locations (buckets);  
    + `output_file` into `output_path`, to be consistent with `input_paths`.  
* Added new CLI command `xcube prune`. The tool deletes all block files associated with empty (NaN-
  only) chunks in given INPUT cube, which must have ZARR format. This can drastically reduce files 
  in sparse cubes and improve cube reading performance. (#92)
* `xcube serve` has a new `prefix` option which is a path appended to the server's host.
  The `prefix` option replaces the `name` option which is now deprecated but kept 
  for backward compatibility. (#79)
* Added new CLI command `xcube resample` that is used to generate temporarily up- or downsampled
  data cubes from other data cubes.
* `xcube serve` can now be run with xcube dataset paths and styling information given via the CLI rather 
  than a configuration file. For example `xcube serve --styles conc_chl=(0,20,"viridis") /path/to/my/chl-cube.zarr`.
  This allows for quick inspection of newly generated cubes via `xcube gen`.
  Also added option `--show` that starts the Xcube viewer on desktop environments in a browser. 
* Added new `xcube apply` command that can be used to generate an output cube computed from a Python function 
  that is applied to one or more input cubes. 
  The command is still in development and therefore hidden.
* Added new `xcube timeit` command that can be used to measure the time required for 
  parameterized command invocations. 
  The command is still in development and therefore hidden.
* Added global `xcube --scheduler SCHEDULER` option for Dask distributed computing (#58)
* Added global `xcube --traceback` option, removed local `xcube gen --traceback` option
* Completed version 1 of an xcube developer guide.
* Added `xcube serve` command (#43) 
* `xcube serve`: Time-series web API now also returns "uncertainty" (#48)
* Added `xcube level` command to allow for creating spatial pyramid levels (#38)
* `xcube gen` accepts multiple configuration files that will be merged in order (#21)
* Added `xcube gen` option `--sort` when input data list should be sorted (#33)    
* Added `xcube vars2dim` command to make variables a cube dimension (#31)
* Added `xcube serve` option `--traceperf` that allows switching on performance diagnostics.
* Included possibility to read the input file paths from a text file. (#47)
* Restructured and clarified code base (#27)
* Moved to Python 3.7 (#25)
* Excluding all input processors except for the default one. They are now plugins and have own repositories within the 
  xcube's organisation. (#49)


### Fixes

* `xcube gen` CLI now updates metadata correctly. (#181)
* It was no longer possible to use the `xcube gen` CLI with `--proc` option. (#120)
* `totalCount` attribute of time series returned by Web API `ts/{dataset}/{variable}/{geom-type}` now
   contains the correct number of possible observations. Was always `1` before.
* Renamed Web API function `ts/{dataset}/{variable}/places` into
  `ts/{dataset}/{variable}/features`.
* `xcube gen` is now taking care that when new time slices are added to an existing
   cube, this is done by maintaining the chronological order. New time slices are
   either appended, inserted, or replaced. (#64) (#139)
* Fixed `xcube serve` issue with WMTS KVP method `GetTile` with query parameter `time` 
  whose value can now also have the two forms `<start-date>/<end-date>` and just `<date>`. (#132) 
* Fixed `xcube extract` regression that stopped working after Pandas update (#95) 
* Fixed problem where CTRL+C didn't function anymore with `xcube serve`. (#87)
* Fixed error `indexes along dimension 'y' are not equal` occurred when using 
  `xcube gen` with processed variables that used flag values (#86)
* Fixed `xcube serve` WMTS KVP API to allow for case-insensitive query parameters. (#77)
* Fixed error in plugins when importing `xcube.api.gen` (#62)
* Fixed import of plugins only when executing `xcube.cli` (#66)

## Changes in 0.1.0

* Respecting chunk sizes when computing tile sizes [#44](https://github.com/dcs4cop/xcube-server/issues/44)
* The RESTful tile operations now have a query parameter `debug=1` which toggles tile 
  computation performance diagnostics.
* Can now associate place groups with datasets.
* Major revision of API. URLs are now more consistent.
* Request for obtaining a legend for a layer of given by a variable of a data set was added.
* Added a Dockerfile to build an xcube docker image and to run the demo
* The RESTful time-series API now returns ISO-formatted UTC dates [#26](https://github.com/dcs4cop/xcube-server/issues/26)
