# The MIT License (MIT)
# Copyright (c) 2022 by the xcube team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import base64
import io
import os
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Dict, Tuple, List, Optional

import fsspec
import matplotlib
import matplotlib.colors
import numpy as np
from PIL import Image
from deprecated import deprecated

from xcube.constants import LOG
from xcube.util.assertions import assert_instance

try:
    # noinspection PyPackageRequirements
    import cmocean.cm as cmo
except ImportError:
    cmo = None

OCEAN_PREFIX = "cmo"
ALPHA_SUFFIX = "_alpha"
REVERSED_SUFFIX = "_r"  # Used by ocm / Ocean

DEFAULT_CMAP_NAME = "viridis"


# Have colormaps separated into categories taken from
# https://matplotlib.org/stable/gallery/color/colormap_reference.html
# colormaps for ocean taken from
# https://matplotlib.org/cmocean/


class ColormapCategory:
    def __init__(self, name: str, desc: str, cm_names: Optional[List[str]] = None):
        self.name = name
        self.desc = desc
        self.cm_names = cm_names


PERCEPTUALLY_UNIFORM_SEQUENTIAL_CATEGORY = ColormapCategory(
    "Perceptually Uniform Sequential",
    "For many applications, a perceptually uniform colormap"
    " is the best choice - one in which equal steps in data are"
    " perceived as equal steps in the color space",
    ["viridis", "plasma", "inferno", "magma", "cividis"],
)

SEQUENTIAL_CATEGORY = ColormapCategory(
    "Sequential",
    "These colormaps are approximately monochromatic colormaps varying"
    " smoothly between two color tones - usually from low saturation"
    " (e.g. white) to high saturation (e.g. a bright blue)."
    " Sequential colormaps are ideal for representing most scientific"
    " data since they show a clear progression from"
    " low-to-high values.",
    [
        "Greys",
        "Purples",
        "Blues",
        "Greens",
        "Oranges",
        "Reds",
        "YlOrBr",
        "YlOrRd",
        "OrRd",
        "PuRd",
        "RdPu",
        "BuPu",
        "GnBu",
        "PuBu",
        "YlGnBu",
        "PuBuGn",
        "BuGn",
        "YlGn",
    ],
)

SEQUENTIAL_2_CATEGORY = ColormapCategory(
    "Sequential (2)",
    "Many of the values from the Sequential 2 plots are monotonically" " increasing.",
    [
        "binary",
        "gist_yarg",
        "gist_gray",
        "gray",
        "bone",
        "pink",
        "spring",
        "summer",
        "autumn",
        "winter",
        "cool",
        "Wistia",
        "hot",
        "afmhot",
        "gist_heat",
        "copper",
    ],
)

DIVERGING_CATEGORY = ColormapCategory(
    "Diverging",
    "These colormaps have a median value (usually light in color) and vary"
    " smoothly to two different color tones at high and low values."
    " Diverging colormaps are ideal when your data has a median value"
    " that is significant (e.g.  0, such that positive and negative"
    " values are represented by different colors of the colormap).",
    [
        "PiYG",
        "PRGn",
        "BrBG",
        "PuOr",
        "RdGy",
        "RdBu",
        "RdYlBu",
        "RdYlGn",
        "Spectral",
        "coolwarm",
        "bwr",
        "seismic",
    ],
)

QUALITATIVE_CATEGORY = ColormapCategory(
    "Qualitative",
    "These colormaps vary rapidly in color."
    " Qualitative colormaps are useful for choosing"
    " a set of discrete colors.",
    [
        "Pastel1",
        "Pastel2",
        "Paired",
        "Accent",
        "Dark2",
        "Set1",
        "Set2",
        "Set3",
        "tab10",
        "tab20",
        "tab20b",
        "tab20c",
    ],
)

CYCLIC_CATEGORY = ColormapCategory(
    "Cyclic", "These colormaps are cyclic.", ["twilight", "twilight_shifted", "hsv"]
)

MISCELLANEOUS_CATEGORY = ColormapCategory(
    "Miscellaneous",
    "Colormaps that don't fit into the categories above"
    " or haven't been categorized yet.",
    [
        "flag",
        "prism",
        "ocean",
        "gist_earth",
        "terrain",
        "gist_stern",
        "gnuplot",
        "gnuplot2",
        "CMRmap",
        "cubehelix",
        "brg",
        "gist_rainbow",
        "rainbow",
        "jet",
        "turbo",
        "nipy_spectral",
        "gist_ncar",
    ],
)

OCEAN_CATEGORY = ColormapCategory(
    "Ocean",
    "Colormaps for commonly-used oceanographic variables" " (from module cmocean.cm).",
)

CUSTOM_CATEGORY = ColormapCategory(
    "Custom",
    "Custom colormaps, e.g. loaded from SNAP *.cpd files.",
)

TEMPLATE_CATEGORIES: Dict[str, ColormapCategory] = {
    c.name: c
    for c in (
        PERCEPTUALLY_UNIFORM_SEQUENTIAL_CATEGORY,
        SEQUENTIAL_CATEGORY,
        SEQUENTIAL_2_CATEGORY,
        DIVERGING_CATEGORY,
        QUALITATIVE_CATEGORY,
        CYCLIC_CATEGORY,
        OCEAN_CATEGORY,
        CUSTOM_CATEGORY,
        MISCELLANEOUS_CATEGORY,
    )
}

_CM_NAME_TO_TEMPLATE_CATEGORY = {
    cm_name: template_cat
    for template_cat in TEMPLATE_CATEGORIES.values()
    if template_cat.cm_names
    for cm_name in template_cat.cm_names
}


class Colormap:
    def __init__(
        self,
        cm_name: str,
        cat_name: Optional[str] = None,
        cmap: Optional[matplotlib.colors.Colormap] = None,
        cmap_reversed: Optional[matplotlib.colors.Colormap] = None,
        cmap_alpha: Optional[matplotlib.colors.Colormap] = None,
        norm: Optional[matplotlib.colors.Normalize] = None,
    ):
        self._cm_name = cm_name
        self._cat_name = cat_name
        self._cmap = cmap
        self._cmap_reversed = cmap_reversed
        self._cmap_alpha = cmap_alpha
        self._cmap_reversed_alpha = None
        self._cmap_png_base64: Optional[str] = None
        self._norm = norm

    @property
    def cm_name(self) -> str:
        return self._cm_name

    @property
    def cat_name(self) -> Optional[str]:
        return self._cat_name

    @cached_property
    def cmap(self) -> matplotlib.colors.Colormap:
        if self._cmap is None:
            self._cmap = matplotlib.colormaps[self.cm_name]
        return self._cmap

    @cached_property
    def cmap_alpha(self) -> matplotlib.colors.Colormap:
        if self._cmap_alpha is None:
            _, self._cmap_alpha = get_alpha_cmap(self.cm_name, self.cmap)
        return self._cmap_alpha

    @cached_property
    def cmap_reversed(self) -> matplotlib.colors.Colormap:
        if self._cmap_reversed is None:
            _, self._cmap_reversed = get_reverse_cmap(self.cm_name, self.cmap)
        return self._cmap_reversed

    @cached_property
    def cmap_reversed_alpha(self) -> matplotlib.colors.Colormap:
        if self._cmap_reversed_alpha is None:
            cm_name, cmap_reversed = get_reverse_cmap(self.cm_name, self.cmap)
            _, self._cmap_reversed_alpha = get_alpha_cmap(cm_name, cmap_reversed)
        return self._cmap_reversed_alpha

    @cached_property
    def cmap_png_base64(self) -> str:
        if self._cmap_png_base64 is None:
            self._cmap_png_base64 = get_cmap_png_base64(self.cmap)
        return self._cmap_png_base64

    @property
    def norm(self) -> Optional[matplotlib.colors.Normalize]:
        return self._norm


class ColormapProvider(ABC):
    @abstractmethod
    def get_cmap(
        self, cm_name: str, num_colors: Optional[int] = None
    ) -> Tuple[str, matplotlib.colors.Colormap]:
        """Get a colormap for the given *cm_name*.

        If *cm_name* is not available, the method may choose another
        colormap and return its name.

        A colormap provider may support colormaps that are

        * reversed (*cm_name* with suffix `"_r"`),
        * have alpha blending (*cm_name* with suffix `"_alpha"`),
        * both (*cm_name* with suffix `"_r_alpha"`).

        Args:
            cm_name: Colormap name.
            num_colors: Optional number of colors, that is, the
                resolution of the colormap gradient.

        Returns:
            A tuple comprising the base name of *cm_name* after striping
            any suffixes, and the colormap as an instance of
            `matplotlib.colors.Colormap`.
        """


class ColormapRegistry(ColormapProvider):
    def __init__(self, *colormaps: Colormap):
        self._categories: Dict[str, ColormapCategory] = {}
        self._colormaps: Dict[str, Colormap] = {}
        # Add standard Matplotlib colormaps
        self._register_mpl_cmaps()
        # Add Ocean colormaps, if any
        self._register_ocm_cmaps()
        # Add custom colormaps, if any
        for colormap in colormaps:
            self.register_colormap(colormap)

    @property
    def categories(self) -> Dict[str, ColormapCategory]:
        return self._categories

    @property
    def colormaps(self) -> Dict[str, Colormap]:
        return self._colormaps

    def register_colormap(self, colormap: Colormap):
        cat_name = colormap.cat_name or MISCELLANEOUS_CATEGORY.name
        template_cat = TEMPLATE_CATEGORIES.get(cat_name)
        if template_cat is None:
            raise ValueError(f"Unknown colormap category {cat_name!r}")
        category = self._get_category(template_cat)
        category.cm_names.append(colormap.cm_name)
        self._colormaps[colormap.cm_name] = colormap

    def find_colormap(self, cm_name: str) -> Optional[Colormap]:
        assert_instance(cm_name, str, name="cm_name")
        # Strips suffixes from cm_name
        cm_name, _, _ = parse_cm_name(cm_name)
        return self._colormaps.get(cm_name)

    def get_cmap(
        self, cm_name: str, num_colors: Optional[int] = None
    ) -> Tuple[str, matplotlib.colors.Colormap]:
        assert_instance(cm_name, str, name="cm_name")
        if num_colors is not None:
            assert_instance(num_colors, int, name="num_colors")

        cm_name, reverse, alpha = parse_cm_name(cm_name)

        colormap = self._colormaps.get(cm_name)
        if colormap is None:
            cm_name = DEFAULT_CMAP_NAME
            colormap = self._colormaps[cm_name]

        if reverse and alpha:
            cmap = colormap.cmap_reversed_alpha
        elif reverse:
            cmap = colormap.cmap_reversed
        elif alpha:
            cmap = colormap.cmap_alpha
        else:
            cmap = colormap.cmap
        if num_colors is not None:
            try:
                # noinspection PyProtectedMember
                cmap = cmap._resample(num_colors)
            except (ValueError, AttributeError, NotImplementedError):
                pass
        return cm_name, cmap

    def to_json(self) -> List:
        result = []
        # Loop through TEMPLATE_CATEGORIES to preserve category order
        for cat_name in TEMPLATE_CATEGORIES.keys():
            category = self._categories.get(cat_name)
            if category is not None:
                result.append(
                    [
                        category.name,
                        category.desc,
                        [
                            [cm_name, self._colormaps[cm_name].cmap_png_base64]
                            for cm_name in category.cm_names
                        ],
                    ]
                )
        return result

    def _get_category(self, template_cat: ColormapCategory):
        category = self._categories.get(template_cat.name)
        if category is None:
            category = ColormapCategory(
                template_cat.name, template_cat.desc, cm_names=[]
            )
            self._categories[template_cat.name] = category
        return category

    def _register_mpl_cmaps(self):
        for cm_name, cmap in matplotlib.colormaps.items():
            if cm_name.startswith(OCEAN_PREFIX):
                # We have a separate category for OCEAN maps
                continue
            if cm_name.endswith(REVERSED_SUFFIX):
                # We register reversed maps, if any, separately
                continue
            template_cat = _CM_NAME_TO_TEMPLATE_CATEGORY.get(cm_name)
            if template_cat is None:
                template_cat = MISCELLANEOUS_CATEGORY
            self.register_colormap(
                Colormap(
                    cm_name,
                    cat_name=template_cat.name,
                    cmap=cmap,
                    cmap_reversed=matplotlib.colormaps.get(cm_name + REVERSED_SUFFIX),
                )
            )

    def _register_ocm_cmaps(self):
        if cmo is None:
            return
        for cm_name in cmo.__dict__.keys():
            if not isinstance(
                getattr(cmo, cm_name), matplotlib.colors.Colormap
            ) or cm_name.endswith(REVERSED_SUFFIX):
                continue
            self.register_colormap(
                Colormap(
                    cm_name,
                    cat_name=OCEAN_CATEGORY.name,
                    cmap=getattr(cmo, cm_name),
                    cmap_reversed=getattr(cmo, cm_name + REVERSED_SUFFIX, None),
                )
            )


def parse_cm_name(cm_name) -> Tuple[str, bool, bool]:
    alpha = cm_name.endswith(ALPHA_SUFFIX)
    if alpha:
        cm_name = cm_name[0 : -len(ALPHA_SUFFIX)]
    reverse = cm_name.endswith(REVERSED_SUFFIX)
    if reverse:
        cm_name = cm_name[0 : -len(REVERSED_SUFFIX)]
    return cm_name, reverse, alpha


def get_reverse_cmap(
    cm_name: str, cmap: matplotlib.colors.Colormap
) -> Tuple[str, matplotlib.colors.Colormap]:
    new_cm_name = cm_name + REVERSED_SUFFIX
    return new_cm_name, cmap.reversed(name=new_cm_name)


def get_alpha_cmap(
    cm_name: str, cmap: matplotlib.colors.Colormap
) -> Tuple[str, matplotlib.colors.Colormap]:
    """Add extra colormaps with alpha gradient.
    See http://matplotlib.org/api/colors_api.html
    """
    new_cm_name = cm_name + ALPHA_SUFFIX

    # noinspection SpellCheckingInspection
    if isinstance(cmap, matplotlib.colors.LinearSegmentedColormap) and hasattr(
        cmap, "_segmentdata"
    ):
        # noinspection PyProtectedMember
        new_segment_data = dict(cmap._segmentdata)
        # let alpha increase from 0.0 to 0.5
        new_segment_data["alpha"] = ((0.0, 0.0, 0.0), (0.5, 1.0, 1.0), (1.0, 1.0, 1.0))
        return new_cm_name, matplotlib.colors.LinearSegmentedColormap(
            new_cm_name, new_segment_data
        )

    if isinstance(cmap, matplotlib.colors.ListedColormap):
        new_colors = list(cmap.colors)
        a_slope = 2.0 / cmap.N
        a = 0
        for i in range(len(new_colors)):
            # noinspection PyUnresolvedReferences
            new_color = new_colors[i]
            if not isinstance(new_color, str):
                if len(new_color) == 3:
                    r, g, b = new_color
                    new_colors[i] = r, g, b, a
                elif len(new_color) == 4:
                    r, g, b, a_old = new_color
                    new_colors[i] = r, g, b, min(a, a_old)
            a += a_slope
            if a > 1.0:
                a = 1.0
        return new_cm_name, matplotlib.colors.ListedColormap(
            new_colors, name=new_cm_name
        )

    LOG.warning(f"Could not create alpha colormap for cmap of type {type(cmap)}")

    return cm_name, cmap


def get_cmap_png_base64(cmap: matplotlib.colors.Colormap) -> str:
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient,))
    image_data = cmap(gradient, bytes=True)
    image = Image.fromarray(image_data, "RGBA")

    # For testing, do:
    # stream = io.FileIO('../cmaps/' + cmap_name + '.png', 'wb')
    # image.save(stream, format='PNG')
    # stream.close()

    stream = io.BytesIO()
    image.save(stream, format="PNG")
    cbar_png_bytes = stream.getvalue()
    stream.close()

    cbar_png_data = base64.b64encode(cbar_png_bytes)
    cbar_png_bytes = cbar_png_data.decode("unicode_escape")

    return cbar_png_bytes


def load_custom_colormap(custom_colormap_path: str) -> Optional[Colormap]:
    try:
        # Currently, we only support SNAP *.cpd files
        colormap = load_snap_cpd_colormap(custom_colormap_path)
        LOG.info(f"Loaded custom colormap" f" {custom_colormap_path!r}")
        return colormap
    except FileNotFoundError:
        LOG.error(f"Missing custom colormap file:" f" {custom_colormap_path}")
    except ValueError as e:
        LOG.error(
            f"Detected invalid custom colormap" f" {custom_colormap_path!r}: {e}",
            exc_info=True,
        )
    return None


def load_snap_cpd_colormap(snap_cpd_path: str) -> Colormap:
    points, log_scaled = _parse_snap_cpd_file(snap_cpd_path)
    samples = [sample for sample, _ in points]
    vmin, vmax = min(samples), max(samples)
    if log_scaled:
        norm = matplotlib.colors.LogNorm(vmin, vmax)
    else:
        norm = matplotlib.colors.Normalize(vmin, vmax)
    colors = list(
        zip(map(norm, samples), ["#%02x%02x%02x" % color for _, color in points])
    )
    cm_name, _ = os.path.splitext(os.path.basename(snap_cpd_path))
    return Colormap(
        cm_name,
        cat_name=CUSTOM_CATEGORY.name,
        cmap=matplotlib.colors.LinearSegmentedColormap.from_list(cm_name, colors),
        norm=norm,
    )


Sample = float
Color = Tuple[int, int, int]
Palette = List[Tuple[Sample, Color]]
LogScaled = bool


def _parse_snap_cpd_file(cpd_file_path: str) -> Tuple[Palette, LogScaled]:
    with fsspec.open(cpd_file_path, mode="r") as f:
        illegal_format_msg = f"Illegal SNAP *.cpd format: {cpd_file_path}"

        entries: Dict[str, str] = dict()
        for line in f.readlines():
            line = line.strip()
            if line and not line.startswith("#"):
                try:
                    k, v = line.split("=", maxsplit=1)
                except ValueError:
                    raise ValueError(illegal_format_msg)
                entries[k.strip()] = v.strip()

        for keyword in ("autoDistribute", "isLogScaled"):
            if keyword in entries:
                LOG.warning(
                    f"Unrecognized keyword {keyword!r}"
                    f" in custom colormap {cpd_file_path}"
                )

        # Uncomment, as soon as we know how to deal with this:
        # log_scaled = entries.get("isLogScaled") == "true"
        log_scaled = False

        try:
            num_points = int(entries.get("numPoints"))
        except ValueError:
            raise ValueError(illegal_format_msg)

        points = []
        for i in range(num_points):
            try:
                r, g, b = map(int, entries.get(f"color{i}", "").split(","))
                sample = float(entries.get(f"sample{i}"))
            except ValueError:
                raise ValueError(illegal_format_msg)
            points.append((sample, (r, g, b)))

        return points, log_scaled


############################################################################
# Deprecated API

_REGISTRY = None
_REGISTRY_JSON = None


@deprecated(
    reason="Colormaps in xcube are no longer managed globally. Use"
    " xcube.util.cmaps.ColormapRegistry.to_json() instead.",
    version="0.13.0",
)
def get_cmaps() -> List:
    """Return a JSON-serializable tuple containing records of the form:
     (<cmap-category>, <cmap-category-description>, <cmap-tuples>),
    where <cmap-tuples> is a tuple containing records of the form
    (<cmap-name>, <cbar-png-bytes>), and where
    <cbar-png-bytes> are encoded PNG images of size 256 x 2 pixels,

    Returns:
        all known matplotlib color maps
    """
    registry = _get_registry()
    global _REGISTRY_JSON
    if _REGISTRY_JSON is None:
        _REGISTRY_JSON = registry.to_json()
    return _REGISTRY_JSON


# noinspection PyUnusedLocal
@deprecated(
    reason="Colormaps in xcube are no longer managed globally. Use"
    " xcube.util.cmaps.ColormapRegistry.get_cmap() instead.",
    version="0.13.0",
)
def get_cmap(
    cmap_name: str, default_cmap_name: str = "viridis", num_colors: Optional[int] = None
) -> Tuple[str, matplotlib.colors.Colormap]:
    """Get color mapping for color bar name *cmap_name*.

    If *num_colors* is a positive integer,
    a resampled mapping will be returned that contains *num_colors*
    color entries.

    If *cmap_name* is not defined, *default_cmap_name* is used.
    If *default_cmap_name* is undefined too, a ValueError is raised.

    Otherwise, a tuple (actual_cmap_name, cmap) is returned.

    Args:
        cmap_name: Color bar name.
        num_colors: Number of colours in returned color mapping.
        default_cmap_name: Default color bar name. (Ignored)

    Returns:
        A tuple (actual_cmap_name, cmap).
    """
    registry = _get_registry()
    return registry.get_cmap(cmap_name, num_colors=num_colors)


@deprecated(
    reason="Colormaps in xcube are no longer managed globally."
    " This function is obsolete.",
    version="0.13.0",
)
def ensure_cmaps_loaded():
    """Loads all color maps from matplotlib and registers additional ones,
    if not done before.
    """
    _get_registry()


def _get_registry() -> ColormapRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = ColormapRegistry()
    return _REGISTRY
