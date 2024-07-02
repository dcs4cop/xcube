# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import unittest
from typing import Any

import pytest
import xarray as xr
import numpy as np

from xcube.core.varexpr import ExprVar
from xcube.core.varexpr import VarExprContext
from xcube.core.varexpr import VarExprError
from xcube.core.varexpr import split_var_assignment

var_a_data = np.array([[0, 1], [2, 3]])
var_b_data = np.array([[1, 2], [3, 4]])
var_c_data = np.array([[1, 0], [0, 1]])
dataset = xr.Dataset(
    dict(
        A=xr.DataArray(var_a_data, dims=("y", "x")),
        B=xr.DataArray(var_b_data, dims=("y", "x")),
        C=xr.DataArray(
            var_c_data,
            dims=("y", "x"),
            attrs={"flag_meanings": "on off", "flag_values": "0, 1"},
        ),
    )
)


class VarExprContextTest(unittest.TestCase):
    def test_namespace(self):

        ctx = VarExprContext(dataset)
        ns = ctx._namespace
        print(list(ns.keys()))

        self.assertIsInstance(ns.get("nan"), float)
        self.assertIsInstance(ns.get("inf"), float)
        self.assertIsInstance(ns.get("e"), float)
        self.assertIsInstance(ns.get("pi"), float)

        self.assertIsInstance(ns.get("A"), ExprVar)
        self.assertIsInstance(ns.get("B"), ExprVar)
        self.assertIsInstance(ns.get("C"), ExprVar)

        self.assertTrue(callable(ns.get("hypot")))

    def test_evaluate(self):
        ctx = VarExprContext(dataset)

        result = ctx.evaluate("A + B")
        self.assertIsInstance(result, xr.DataArray)
        np.testing.assert_equal(result.values, var_a_data + var_b_data)

        result = ctx.evaluate("hypot(A, B)")
        self.assertIsInstance(result, xr.DataArray)
        np.testing.assert_equal(
            result.values,
            np.hypot(var_a_data, var_b_data),
        )

        result = ctx.evaluate("(A <= 2) & (B >= 3)")
        self.assertIsInstance(result, xr.DataArray)
        np.testing.assert_equal(result.values, (var_a_data <= 2) & (var_b_data >= 3))

    def test_syntax_errors(self):
        ctx = VarExprContext(dataset)
        self.assert_syntax_error(ctx, "x = A")
        self.assert_syntax_error(ctx, "A + 1; B + 2")

    # noinspection PyMethodMayBeStatic
    @staticmethod
    def assert_syntax_error(ctx: VarExprContext, expr: str):
        with pytest.raises(VarExprError, match="invalid syntax"):
            ctx.evaluate(expr)

    # noinspection PyMethodMayBeStatic
    def test_invalid_expressions(self):
        ctx = VarExprContext(dataset)

        with pytest.raises(
            VarExprError,
            match="'DataArray' object has no attribute 'dims'",
        ):
            ctx.evaluate("A.dims")

        with pytest.raises(
            VarExprError,
            match="result must be a 'DataArray' object, but got type 'int'",
        ):
            ctx.evaluate("137")

        with pytest.raises(
            VarExprError,
            match="result must be a 'DataArray' object, but got type 'tuple'",
        ):
            ctx.evaluate("A, B")

    # noinspection PyMethodMayBeStatic
    def test_intentional_disintegration(self):
        ctx = VarExprContext(dataset)
        # This is intentional disintegration
        ev = ExprVar(xr.DataArray(var_a_data, dims=("y", "x")))
        ev.__dict__["_ExprVar__da"] = True
        ctx._namespace["my_ev"] = ev
        with pytest.raises(
            RuntimeError,
            match="internal error: 'DataArray' object expected, but got type 'bool'",
        ):
            ctx.evaluate("my_ev")


class ExprVarTest(unittest.TestCase):

    # noinspection PyMethodMayBeStatic
    def test_ctor_raises(self):
        with pytest.raises(TypeError):
            # noinspection PyTypeChecker
            ExprVar(137)

    def test_supported_ops(self):
        da1 = xr.DataArray([1, 2, 3], dims="x")
        da2 = xr.DataArray([2, 3, 4], dims="x")
        ev1 = ExprVar(da1)
        ev2 = ExprVar(da2)

        # Binary operations - comparisons

        self.assert_ev(ev1 == ev2, [False, False, False])
        self.assert_ev(ev1 == 1, [True, False, False])

        self.assert_ev(ev1 != ev2, [True, True, True])
        self.assert_ev(ev1 != 1, [False, True, True])

        self.assert_ev(ev1 <= ev2, [True, True, True])
        self.assert_ev(ev1 <= 1, [True, False, False])

        self.assert_ev(ev1 < ev2, [True, True, True])
        self.assert_ev(ev1 < 1, [False, False, False])

        self.assert_ev(ev1 >= ev2, [False, False, False])
        self.assert_ev(ev1 >= 1, [True, True, True])

        self.assert_ev(ev1 > ev2, [False, False, False])
        self.assert_ev(ev1 > 1, [False, True, True])

        # Binary operations - emulating numeric type

        self.assert_ev(ev1 + ev2, [3, 5, 7])
        self.assert_ev(ev1 + 3, [4, 5, 6])
        self.assert_ev(1 + ev2, [3, 4, 5])

        self.assert_ev(ev1 - ev2, [-1, -1, -1])
        self.assert_ev(ev1 - 1, [0, 1, 2])
        self.assert_ev(5 - ev2, [3, 2, 1])

        self.assert_ev(ev1 * ev2, [2, 6, 12])
        self.assert_ev(ev1 * 2, [2, 4, 6])
        self.assert_ev(3 * ev2, [6, 9, 12])

        self.assert_ev(ev2 / ev1, [2.0, 1.5, 4 / 3])
        self.assert_ev(ev1 / 2, [0.5, 1.0, 1.5])
        self.assert_ev(3 / ev2, [1.5, 1.0, 0.75])

        self.assert_ev(ev2 // ev1, [2, 1, 1])
        self.assert_ev(ev1 // 2, [0, 1, 1])
        self.assert_ev(3 // ev2, [1, 1, 0])

        self.assert_ev(ev2 % ev1, [0, 1, 1])
        self.assert_ev(ev1 % 2, [1, 0, 1])
        self.assert_ev(3 % ev2, [1, 0, 3])

        self.assert_ev(ev1**ev2, [1, 8, 81])
        self.assert_ev(ev1**3, [1, 8, 27])
        self.assert_ev(2**ev2, [4, 8, 16])

        self.assert_ev(ev1 << ev2, [4, 16, 48])
        self.assert_ev(ev1 << 1, [2, 4, 6])

        self.assert_ev(ev1 >> ev2, [0, 0, 0])
        self.assert_ev(ev1 >> 1, [0, 1, 1])

        self.assert_ev(ev1 & ev2, [0, 2, 0])
        self.assert_ev(ev1 & 3, [1, 2, 3])
        self.assert_ev(2 & ev2, [2, 2, 0])

        self.assert_ev(ev1 ^ ev2, [3, 1, 7])
        self.assert_ev(ev1 ^ 3, [2, 1, 0])
        self.assert_ev(2 ^ ev2, [0, 1, 6])

        self.assert_ev(ev1 | ev2, [3, 3, 7])
        self.assert_ev(ev1 | 3, [3, 3, 3])
        self.assert_ev(2 | ev2, [2, 3, 6])

        # Unary operations

        self.assert_ev(+ev1, [1, 2, 3])
        self.assert_ev(-ev1, [-1, -2, -3])
        self.assert_ev(~ev1, [-2, -3, -4])

    def assert_ev(self, ev: Any, expected_result: list):
        self.assertIsInstance(ev, ExprVar)
        da = ev.__dict__["_ExprVar__da"]
        self.assertIsInstance(da, xr.DataArray)
        self.assertEqual(expected_result, list(da.values))

    # noinspection PyMethodMayBeStatic
    def test_that_data_array_is_not_easily_accessible(self):
        da = xr.DataArray([1, 2, 3], dims="x")
        ev = ExprVar(da)

        with pytest.raises(
            AttributeError,
            match="'ExprVar' object has no attribute '_ExprVarTest__da'",
        ):
            # noinspection PyUnusedLocal
            result = ev.__da

    # noinspection PyMethodMayBeStatic
    def test_that_some_ops_unsupported(self):
        ev = ExprVar(xr.DataArray([1, 2, 3], dims="x"))

        with pytest.raises(
            TypeError,
            match="unsupported operand type\\(s\\) for <<: 'int' and 'DataArray'",
        ):
            # noinspection PyUnusedLocal
            result = 1 << ev

        with pytest.raises(
            TypeError,
            match="unsupported operand type\\(s\\) for >>: 'int' and 'DataArray'",
        ):
            # noinspection PyUnusedLocal
            result = 1 >> ev


class HelpersTest(unittest.TestCase):
    def test_split_var_assignment(self):
        self.assertEqual(("A", None), split_var_assignment("A"))
        self.assertEqual(("A", "B"), split_var_assignment("A=B"))
        self.assertEqual(("A", "B + C"), split_var_assignment(" A = B + C "))
