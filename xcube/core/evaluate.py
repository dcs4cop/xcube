# The MIT License (MIT)
# Copyright (c) 2019 by the xcube development team and contributors
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

import functools
import math

import numpy as np
import xarray as xr

from xcube.core.maskset import MaskSet
from xcube.util.config import NameDictPairList
from xcube.util.config import to_resolved_name_dict_pairs
from xcube.util.expression import compute_array_expr


def evaluate_dataset(
    dataset: xr.Dataset,
    processed_variables: NameDictPairList = None,
    errors: str = "raise",
) -> xr.Dataset:
    """Compute new variables or mask existing variables in *dataset*
    by the evaluation of Python expressions, that may refer to other
    existing or new variables.
    Returns a new dataset that contains the old and new variables,
    where both may bew now masked.

    Expressions may be given by attributes of existing variables in
    *dataset* or passed a via the *processed_variables* argument
    which is a sequence of variable name / attributes tuples.

    Two types of expression attributes are recognized in the attributes:

    1. The attribute `expression` generates
       a new variable computed from its attribute value.
    2. The attribute `valid_pixel_expression` masks out
       invalid variable values.

    In both cases the attribute value must be a string that forms
    a valid Python expression that can reference any other preceding
    variables by name.
    The expression can also reference any flags defined by another
    variable according to their CF attributes `flag_meaning`
    and `flag_values`.

    Invalid variable values may be masked out using the value the
    `valid_pixel_expression` attribute whose value should form
    a Boolean Python expression. In case, the expression
    returns zero or false, the value of the `_FillValue` attribute
    or NaN will be used in the new variable.

    Other attributes will be stored as variable metadata as-is.

    Args:
        dataset: A dataset.
        processed_variables: Optional list of variable name-attributes
            pairs that will processed in the given order.
        errors: How to deal with errors while evaluating expressions.
            May be be one of "raise", "warn", or "ignore".

    Returns:
        new dataset with computed variables
    """

    if processed_variables:
        processed_variables = to_resolved_name_dict_pairs(
            processed_variables, dataset, keep=True
        )
    else:
        var_names = list(dataset.data_vars)
        var_names = sorted(var_names, key=functools.partial(_get_var_sort_key, dataset))
        processed_variables = [(var_name, None) for var_name in var_names]

    # Initialize namespace with some constants and modules
    namespace = dict(NaN=np.nan, PI=math.pi, np=np, xr=xr)
    # Now add all mask sets and variables
    for var_name in dataset.data_vars:
        var = dataset[var_name]
        if MaskSet.is_flag_var(var):
            namespace[var_name] = MaskSet(var)
        else:
            namespace[var_name] = var

    for var_name, var_props in processed_variables:
        if var_name in dataset.data_vars:
            # Existing variable
            var = dataset[var_name]
            if var_props:
                var_props_temp = var_props
                var_props = dict(var.attrs)
                var_props.update(var_props_temp)
            else:
                var_props = dict(var.attrs)
        else:
            # Computed variable
            var = None
            if var_props is None:
                var_props = dict()

        do_load = var_props.get("load", False)

        expression = var_props.get("expression")
        if expression:
            # Compute new variable
            computed_array = compute_array_expr(
                expression,
                namespace=namespace,
                result_name=f"{var_name!r}",
                errors=errors,
            )
            if computed_array is not None:
                if hasattr(computed_array, "attrs"):
                    var = computed_array
                    var.attrs.update(var_props)
                if do_load:
                    computed_array.load()
                namespace[var_name] = computed_array

        valid_pixel_expression = var_props.get("valid_pixel_expression")
        if valid_pixel_expression:
            # Compute new mask for existing variable
            if var is None:
                raise ValueError(f"undefined variable {var_name!r}")
            valid_mask = compute_array_expr(
                valid_pixel_expression,
                namespace=namespace,
                result_name=f"valid mask for {var_name!r}",
                errors=errors,
            )
            if valid_mask is not None:
                masked_var = var.where(valid_mask)
                if hasattr(masked_var, "attrs"):
                    masked_var.attrs.update(var_props)
                if do_load:
                    masked_var.load()
                namespace[var_name] = masked_var

    computed_dataset = dataset.copy()
    for name, value in namespace.items():
        if isinstance(value, xr.DataArray):
            computed_dataset[name] = value

    return computed_dataset


def _get_var_sort_key(dataset: xr.Dataset, var_name: str):
    # noinspection SpellCheckingInspection
    attrs = dataset[var_name].attrs
    a1 = attrs.get("expression")
    a2 = attrs.get("valid_pixel_expression")
    v1 = 10 * len(a1) if a1 is not None else 0
    v2 = 100 * len(a2) if a2 is not None else 0
    return v1 + v2
