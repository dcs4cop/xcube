# The MIT License (MIT)
# Copyright (c) 2019-2021 by the xcube development team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
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

from typing import Dict, Any, Iterable

import dask.array as da
import numpy as np
import xarray as xr


# TODO: this would be useful to have in xarray:
#       >>> ds = xr.open_dataset("my/path/to/cf/netcdf.nc", decode_flags=True)
#       >>> ds.flag_mask_sets['quality_flags']


class MaskSet:
    """A set of mask variables derived from a variable *flag_var* with the following
    CF attributes:

      - One or both of `flag_masks` and `flag_values`
      - `flag_meanings` (always required)

    See https://cfconventions.org/Data/cf-conventions/cf-conventions-1.9/cf-conventions.html#flags
    for details on the use of these attributes.

    Each mask is represented by an `xarray.DataArray`, has the name of the flag,
    is of type `numpy.unit8`, and has the dimensions of the given *flag_var*.

    Args:
        flag_var: an `xarray.DataArray` that defines flag values. The CF
            attributes `flag_meanings` and one or both of `flag_masks`
            and `flag_values` are expected to exist and be valid.
    """

    def __init__(self, flag_var: xr.DataArray):
        flag_masks = flag_var.attrs.get("flag_masks")
        flag_values = flag_var.attrs.get("flag_values")
        if flag_masks is None and flag_values is None:
            raise ValueError(
                "One or both of the attributes "
                "'flag_masks' or 'flag_values' "
                "must be present and non-null in flag_var"
            )
        if flag_masks is not None:
            flag_masks = _convert_flag_var_attribute_value(flag_masks, "flag_masks")
        if flag_values is not None:
            flag_values = _convert_flag_var_attribute_value(flag_values, "flag_values")
        if "flag_meanings" not in flag_var.attrs:
            raise ValueError("flag_var must have the attribute 'flag_meanings'")
        flag_meanings = flag_var.attrs.get("flag_meanings")
        if not isinstance(flag_meanings, str):
            raise TypeError("attribute 'flag_meanings' of flag_var " "must be a string")
        flag_names = flag_meanings.split(" ")
        if flag_masks is not None and len(flag_names) != len(flag_masks):
            raise ValueError(
                "attributes 'flag_meanings' and 'flag_masks' " "are not corresponding"
            )
        if flag_values is not None and len(flag_names) != len(flag_values):
            raise ValueError(
                "attributes 'flag_meanings' and 'flag_values' " "are not corresponding"
            )
        if flag_masks is None:
            flag_masks = [None] * len(flag_names)
        if flag_values is None:
            flag_values = [None] * len(flag_names)
        self._flag_var = flag_var
        self._flag_names = flag_names
        self._flags = dict(zip(flag_names, list(zip(flag_masks, flag_values))))
        self._masks = {}

    @classmethod
    def is_flag_var(cls, var: xr.DataArray) -> bool:
        return "flag_meanings" in var.attrs and (
            "flag_masks" in var.attrs or "flag_values" in var.attrs
        )

    @classmethod
    def get_mask_sets(cls, dataset: xr.Dataset) -> Dict[str, "MaskSet"]:
        """For each "flag" variable in given *dataset*, turn it into a `MaskSet`,
        store it in a dictionary.

        Args:
            dataset: The dataset

        Returns:
            A mapping of flag names to `MaskSet`. Will be empty if
            there are no flag variables in *dataset*.
        """
        masks = {}
        for var_name in dataset.variables:
            var = dataset[var_name]
            if cls.is_flag_var(var):
                masks[var_name] = MaskSet(var)
        return masks

    def _repr_html_(self):
        lines = [
            "<html>",
            "<table>",
            "<tr><th>Flag name</th><th>Mask</th><th>Value</th></tr>",
        ]

        for name, data in self._flags.items():
            mask, value = data
            lines.append(f"<tr><td>{name}</td><td>{mask}</td><td>{value}</td></tr>")

        lines.extend(["</table>", "</html>"])

        return "\n".join(lines)

    def __str__(self):
        return "%s(%s)" % (
            self._flag_var.name,
            ", ".join(["%s=%s" % (n, v) for n, v in self._flags.items()]),
        )

    def __dir__(self) -> Iterable[str]:
        return self._flag_names

    def __getattr__(self, name: str) -> Any:
        if name not in self._flags:
            raise AttributeError(name)
        return self.get_mask(name)

    def __getitem__(self, item):
        try:
            name = self._flag_names[item]
            if name not in self._flags:
                raise IndexError(item)
        except TypeError:
            name = item
            if name not in self._flags:
                raise KeyError(item)
        return self.get_mask(name)

    def __contains__(self, item):
        return item in self._flags

    def get_mask(self, flag_name: str):
        if flag_name not in self._flags:
            raise ValueError('invalid flag name "%s"' % flag_name)

        if flag_name in self._masks:
            return self._masks[flag_name]

        flag_var = self._flag_var
        flag_mask, flag_value = self._flags[flag_name]

        if flag_var.chunks is not None:
            ones_array = da.ones(flag_var.shape, dtype=np.uint8, chunks=flag_var.chunks)
        else:
            ones_array = np.ones(flag_var.shape, dtype=np.uint8)

        mask_var = xr.DataArray(
            ones_array, dims=flag_var.dims, name=flag_name, coords=flag_var.coords
        )
        if flag_mask is not None:
            if flag_var.dtype != flag_mask.dtype:
                flag_var = flag_var.astype(flag_mask.dtype)
            if flag_value is not None:
                mask_var = mask_var.where((flag_var & flag_mask) == flag_value, 0)
            else:
                mask_var = mask_var.where((flag_var & flag_mask) != 0, 0)
        else:
            if flag_var.dtype != flag_value.dtype:
                flag_var = flag_var.astype(flag_value.dtype)
            mask_var = mask_var.where(flag_var == flag_value, 0)

        self._masks[flag_name] = mask_var
        return mask_var


_MASK_DTYPES = (
    (2**8, np.uint8),
    (2**16, np.uint16),
    (2**32, np.uint32),
    (2**64, np.uint64),
)


def _convert_flag_var_attribute_value(attr_value, attr_name):
    if isinstance(attr_value, str):
        err_msg = f'Invalid bit expression in value for {attr_name}: "{attr_value}"'
        masks = []
        max_mask = 0
        for s in attr_value.split(","):
            s = s.strip()
            pair = s.split("-")
            if len(pair) == 1:
                try:
                    mask = (1 << int(s[0:-1])) if s.endswith("b") else int(s)
                except ValueError as e:
                    raise ValueError(err_msg) from e
            elif len(pair) == 2:
                s1, s2 = pair
                if not s1.endswith("b") or not s2.endswith("b"):
                    raise ValueError(err_msg)
                try:
                    b1 = int(s1[0:-1])
                    b2 = int(s2[0:-1])
                except ValueError as e:
                    raise ValueError(err_msg) from e
                if b1 > b2:
                    raise ValueError(err_msg)
                mask = 0
                for b in range(b1, b2 + 1):
                    mask |= 1 << b
            else:
                raise ValueError(err_msg)
            masks.append(mask)
            max_mask = max(max_mask, mask)

        for limit, dtype in _MASK_DTYPES:
            if max_mask <= limit:
                return np.array(masks, dtype)

        raise ValueError(err_msg)

    if not (hasattr(attr_value, "dtype") and hasattr(attr_value, "shape")):
        raise TypeError(f"attribute {attr_name!r} must be an integer array")

    return attr_value
