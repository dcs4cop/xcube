# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from typing import Callable, Optional

import numpy as np
import xarray as xr

from xcube.util.assertions import assert_instance


class ExprVar:
    """A wrapped `xarray.DataArray` to allow safe access in expressions.

    Args:
        da: The `xarray.DataArray` to be wrapped.
    """

    def __init__(self, da: xr.DataArray):
        assert_instance(da, xr.DataArray, name="v")
        # Note that the double underscore protects access by "name mangling"
        self.__da = da

    ################################################
    # Binary operations - comparisons

    def __eq__(self, other):
        return self.__wrap(self.__da == self.__unwrap(other))

    def __ne__(self, other):
        return self.__wrap(self.__da != self.__unwrap(other))

    def __le__(self, other):
        return self.__wrap(self.__da <= self.__unwrap(other))

    def __lt__(self, other):
        return self.__wrap(self.__da < self.__unwrap(other))

    def __ge__(self, other):
        return self.__wrap(self.__da >= self.__unwrap(other))

    def __gt__(self, other):
        return self.__wrap(self.__da > self.__unwrap(other))

    ################################################
    # Binary operations - emulating numeric type

    def __add__(self, other):
        return self.__wrap(self.__da + self.__unwrap(other))

    def __radd__(self, other):
        return self.__wrap(self.__unwrap(other) + self.__da)

    def __sub__(self, other):
        return self.__wrap(self.__da - self.__unwrap(other))

    def __rsub__(self, other):
        return self.__wrap(self.__unwrap(other) - self.__da)

    def __mul__(self, other):
        return self.__wrap(self.__da * self.__unwrap(other))

    def __rmul__(self, other):
        return self.__wrap(self.__unwrap(other) * self.__da)

    def __truediv__(self, other):
        return self.__wrap(self.__da / self.__unwrap(other))

    def __rtruediv__(self, other):
        return self.__wrap(self.__unwrap(other) / self.__da)

    def __floordiv__(self, other):
        return self.__wrap(self.__da // self.__unwrap(other))

    def __rfloordiv__(self, other):
        return self.__wrap(self.__unwrap(other) // self.__da)

    def __mod__(self, other):
        return self.__wrap(self.__da % self.__unwrap(other))

    def __rmod__(self, other):
        return self.__wrap(self.__unwrap(other) % self.__da)

    def __pow__(self, power):
        return self.__wrap(self.__da ** self.__unwrap(power))

    def __rpow__(self, other):
        return self.__wrap(self.__unwrap(other) ** self.__da)

    def __lshift__(self, other):
        return self.__wrap(self.__da << self.__unwrap(other))

    def __rlshift__(self, other):
        # Not supported by xarray, will raise
        return self.__wrap(self.__unwrap(other) << self.__da)

    def __rshift__(self, other):
        return self.__wrap(self.__da >> self.__unwrap(other))

    def __rrshift__(self, other):
        # Not supported by xarray, will raise
        return self.__wrap(self.__unwrap(other) >> self.__da)

    def __and__(self, other):
        return self.__wrap(self.__da & self.__unwrap(other))

    def __rand__(self, other):
        return self.__wrap(self.__unwrap(other) & self.__da)

    def __xor__(self, other):
        return self.__wrap(self.__da ^ self.__unwrap(other))

    def __rxor__(self, other):
        return self.__wrap(self.__unwrap(other) ^ self.__da)

    def __or__(self, other):
        return self.__wrap(self.__da | self.__unwrap(other))

    def __ror__(self, other):
        return self.__wrap(self.__unwrap(other) | self.__da)

    ################################################
    # Unary operations

    def __pos__(self):
        return self.__wrap(+self.__da)

    def __neg__(self):
        return self.__wrap(-self.__da)

    def __invert__(self):
        return self.__wrap(~self.__da)

    ################################################
    # Internal helpers

    @staticmethod
    def __unwrap(v):
        return v.__da if isinstance(v, ExprVar) else v

    @staticmethod
    def __wrap(v):
        return ExprVar(v) if isinstance(v, xr.DataArray) else v

    @staticmethod
    def _wrap_fn(fn: Callable) -> Callable:
        def wrapped_fn(*args, **kwargs):
            return ExprVar.__wrap(
                fn(
                    *(ExprVar.__unwrap(arg) for arg in args),
                    **{kw: ExprVar.__unwrap(arg) for kw, arg in kwargs.items()},
                )
            )

        return wrapped_fn


def _get_safe_xarray_funcs() -> dict[str, Callable]:
    # noinspection PyProtectedMember
    return dict(where=ExprVar._wrap_fn(xr.where))


def _get_safe_numpy_funcs() -> dict[str, Callable]:
    # noinspection PyProtectedMember
    return {
        k: ExprVar._wrap_fn(v)
        for k, v in np.__dict__.items()
        if isinstance(v, np.ufunc) and isinstance(k, str) and not k.startswith("_")
    }


# noinspection PyProtectedMember
_BASE_NAMESPACE = dict(
    nan=np.nan,
    e=np.e,
    inf=np.inf,
    pi=np.pi,
    **_get_safe_xarray_funcs(),
    **_get_safe_numpy_funcs(),
)


class VarExprError(ValueError):
    """Exception raised by the `VarExprContext` class."""


class VarExprContext:
    """Allow safe evaluation of expressions in the context of a
    `xarray.Dataset` object.

    Args:
        dataset: The dataset that provides the variables that can
            be accessed in the expressions passed to :meth:`evaluate`.
    """

    def __init__(self, dataset: xr.Dataset):
        namespace = dict(_BASE_NAMESPACE)
        namespace.update({str(k): ExprVar(v) for k, v in dataset.data_vars.items()})
        namespace.update({str(k): ExprVar(v) for k, v in dataset.coords.items()})
        self._namespace = namespace

    def evaluate(self, var_expr: str) -> xr.DataArray:
        """Evaluate given Python expression *var_expr* in the context of an
        `xarray.Dataset` object.

        The expression *var_expr* may reference the following names:

        * the dataset's data variables;
        * the dataset's coordinate variables;
        * the numpy constants `e`, `pi`, `nan`, `inf`;
        * all numpy ufuncs (https://numpy.org/doc/stable/reference/ufuncs.html);
        * the `where` function (https://docs.xarray.dev/en/stable/generated/xarray.where.html).

        In general, all Python numerical and logical operators such as
        `not`, `and`, `or` are supported.
        However, for dataset variables the following subset of operators apply:

        * binary comparison operators: `==`, `!=`, `<`, `<=`, `>`, `>=`;
        * binary arithmetic operators: `+`, `-`, `*`, `/`, `//`, `%`,
            `**`, `<<`, `>>`, `&`, `^`, `|`;
        * unary operators: `+`, `-`, `~`.

        Args:
            var_expr: Expression to be evaluated.

        Returns:
            A newly computed variable of type `xarray.DataArray`.
        """
        try:
            result = eval(var_expr, self._namespace, None)
        except BaseException as e:
            # Do not report the name 'ExprVar'
            raise VarExprError(f"{e}".replace("ExprVar", "DataArray")) from e

        if not isinstance(result, ExprVar):
            # We do not mention 'ExprVar' by intention
            raise VarExprError(
                f"result must be a 'DataArray' object,"
                f" but got type {result.__class__.__name__!r}"
            )

        result = result.__dict__.get("_ExprVar__da")
        if not isinstance(result, xr.DataArray):
            # noinspection PyUnresolvedReferences
            raise RuntimeError(
                f"internal error: 'DataArray' object expected,"
                f" but got type {result.__class__.__name__!r}"
            )

        return result


def split_var_assignment(var_name_or_assign: str) -> tuple[str, Optional[str]]:
    """Split *var_name_or_assign* into a variable name and expression part.

    Args:
        var_name_or_assign: A variable name or an expression

    Return:
        A pair (var_name, var_expr) if *var_name_or_assign* is an assignment
        expression, otherwise (var_name, None).
    """
    if "=" in var_name_or_assign:
        var_name, var_expr = map(
            lambda s: s.strip(), var_name_or_assign.split("=", maxsplit=1)
        )
        return var_name, var_expr
    else:
        return var_name_or_assign, None
