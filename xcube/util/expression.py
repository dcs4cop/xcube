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

import ast
import warnings
from typing import Dict, Any


def compute_array_expr(
    expr: str,
    namespace: Dict[str, Any] = None,
    errors: str = "raise",
    result_name: str = None,
):
    """Safely evaluate a Python expression and return the result.

    The namespace is expected to contain variable references
    which, when evaluated, return numpy-like data arrays.

    For example, the Python expression
    "B01 <= 0 and not min(B01) > 2" contains the variable reference
    "B01". The expression is first transpiled to the new expression
    "np.logical_and(B01 <= 0, np.logical_not(np.fmin(B01) > 2))"
    which is then being evaluated instead.

    Args:
        expr: A valid Python expression.
        namespace: A dictionary that represents the namespace for the
            evaluation.
        result_name: Name of the result (used for error messages only)
        errors: How to deal with errors raised when evaluating the
            expression. May be one of "raise" or "warn".

    Returns:
        The result computed from the evaluating the expression.
    """
    expr = transpile_expr(expr, warn=errors == "warn")
    return compute_expr(
        expr, namespace=namespace, errors=errors, result_name=result_name
    )


def compute_expr(
    expr: str,
    namespace: Dict[str, Any] = None,
    errors: str = "raise",
    result_name: str = None,
):
    """Safely evaluate a Python expression and return the result.

    Args:
        expr: A valid Python expression.
        namespace: A dictionary that represents the namespace for the
            evaluation.
        result_name: Name of the result (used for error messages only)
        errors: How to deal with errors raised when evaluating the
            expression. May be one of "raise" or "warn".

    Returns:
        The result computed from the evaluating the expression.
    """
    try:
        result = eval(expr, namespace, None)
    except Exception as e:
        result = None
        if result_name:
            msg = f"failed computing {result_name}" f" from expression {expr!r}: {e}"
        else:
            msg = f"failed computing expression {expr!r}: {e}"
        if errors == "raise":
            raise ValueError(msg) from e
        if errors == "warn":
            warnings.warn(msg)
    return result


def transpile_expr(expr: str, warn=False) -> str:
    """Transpile a Python expression into a numpy/xarray array expression.

    Args:
        expr: The Python expression
        warn: If warnings shall be emitted
    Returns:
        The numpy array expression
    """
    return _ExprTranspiler(expr, warn).transpile()


# noinspection PyMethodMayBeStatic
class _ExprTranspiler:
    """Transpile a Python expression into a numpy/xarray array expression.

    See https://greentreesnakes.readthedocs.io/en/latest/nodes.html#expressions
    """

    _KEYWORDS = {
        "in",
        "not in",
        "is",
        "is not",
        "and",
        "or",
        "not",
        "True",
        "False",
        "None",
    }

    _OP_INFOS = {
        ast.IfExp: ("if", 10, "L"),
        ast.Eq: ("==", 100, "R"),
        ast.NotEq: ("!=", 100, "R"),
        ast.Lt: ("<", 100, "R"),
        ast.LtE: ("<=", 100, "R"),
        ast.Gt: (">", 100, "R"),
        ast.GtE: (">=", 100, "R"),
        ast.Is: ("is", 100, "R"),
        ast.IsNot: ("is not", 100, "R"),
        ast.In: ("in", 100, "R"),
        ast.NotIn: ("not in", 100, "R"),
        ast.Or: ("or", 300, "L"),
        ast.And: ("and", 400, "L"),
        ast.Not: ("not", 500, None),
        ast.UAdd: ("+", 600, None),
        ast.USub: ("-", 600, None),
        ast.Add: ("+", 600, "E"),
        ast.Sub: ("-", 600, "L"),
        ast.Mult: ("*", 700, "E"),
        ast.Div: ("/", 700, "L"),
        ast.FloorDiv: ("//", 700, "L"),
        ast.Mod: ("%", 800, "L"),
        ast.Pow: ("**", 900, "L"),
    }

    @classmethod
    def get_op_info(cls, op: ast.AST):
        return cls._OP_INFOS.get(type(op), (None, None, None))

    def __init__(self, expr: str, warn: bool):
        self.expr = expr
        self.warn = warn

    def transpile(self) -> str:
        return self._transpile(ast.parse(self.expr))

    def _transpile(self, node: ast.AST) -> str:
        if isinstance(node, ast.Module):
            return self._transpile(node.body[0])
        if isinstance(node, ast.Expr):
            return self._transpile(node.value)
        if isinstance(node, ast.Name):
            return self.transform_name(node)
        if isinstance(node, ast.NameConstant):
            return self.transform_name_constant(node)
        if isinstance(node, ast.Num):
            return self.transform_num(node)
        if isinstance(node, ast.Attribute):
            pat = self.transform_attribute(node.value, node.attr, node.ctx)
            x = self._transpile(node.value)
            return pat.format(x=x)
        if isinstance(node, ast.Call):
            pat = self.transform_call(node.func, node.args)
            xes = {
                "x%s" % i: self._transpile(node.args[i]) for i in range(len(node.args))
            }
            return pat.format(**xes)
        if isinstance(node, ast.UnaryOp):
            pat = self.transform_unary_op(node.op, node.operand)
            arg = self._transpile(node.operand)
            return pat.format(x=arg)
        if isinstance(node, ast.BinOp):
            pat = self.transform_bin_op(node.op, node.left, node.right)
            x = self._transpile(node.left)
            y = self._transpile(node.right)
            return pat.format(x=x, y=y)
        if isinstance(node, ast.IfExp):
            pat = self.transform_if_exp(node.test, node.body, node.orelse)
            x = self._transpile(node.test)
            y = self._transpile(node.body)
            z = self._transpile(node.orelse)
            return pat.format(x=x, y=y, z=z)
        if isinstance(node, ast.BoolOp):
            pat = self.transform_bool_op(node.op, node.values)
            xes = {
                "x%s" % i: self._transpile(node.values[i])
                for i in range(len(node.values))
            }
            return pat.format(**xes)
        if isinstance(node, ast.Compare):
            pat = self.transform_compare(node.left, node.ops, node.comparators)
            xes = {"x0": self._transpile(node.left)}
            xes.update(
                {
                    "x%s" % (i + 1): self._transpile(node.comparators[i])
                    for i in range(len(node.comparators))
                }
            )
            return pat.format(**xes)
        raise ValueError(
            'unrecognized expression node %s in "%s"'
            % (node.__class__.__name__, self.expr)
        )

    def transform_name(self, name: ast.Name):
        return name.id

    def transform_name_constant(self, node: ast.NameConstant):
        return repr(node.value)

    def transform_num(self, node: ast.Num):
        return str(node.n)

    def transform_call(self, func: ast.Name, args):
        args = ", ".join(["{x%d}" % i for i in range(len(args))])
        return "%s(%s)" % (self.transform_function_name(func), args)

    def transform_function_name(self, func: ast.Name):
        if isinstance(func, ast.Attribute):
            return "%s.%s" % (func.value.id, func.attr)
        else:
            if func.id in ("where",):
                return "xr.%s" % func.id
            else:
                name = dict(min="fmin", max="fmax").get(func.id, func.id)
                return "np.%s" % name

    # noinspection PyUnusedLocal
    def transform_attribute(self, value: ast.AST, attr: str, ctx):
        return "{x}.%s" % attr

    def transform_unary_op(self, op, operand):
        name, precedence, _ = self.get_op_info(op)

        x = "{x}"

        if name == "not":
            return "np.logical_not(%s)" % x

        right_op = getattr(operand, "op", None)
        if right_op:
            _, other_precedence, other_assoc = self.get_op_info(right_op)
            if other_precedence < precedence or (
                other_precedence == precedence and other_assoc is not None
            ):
                x = "({x})"

        if name in self._KEYWORDS:
            return "%s %s" % (name, x)
        else:
            return "%s%s" % (name, x)

    def transform_bin_op(self, op, left, right):
        name, precedence, assoc = _ExprTranspiler.get_op_info(op)

        x = "{x}"
        y = "{y}"

        if name == "**":
            return "np.power(%s, %s)" % (x, y)

        left_op = getattr(left, "op", None)
        right_op = getattr(right, "op", None)

        if left_op:
            _, other_precedence, other_assoc = self.get_op_info(left_op)
            if other_precedence < precedence or (
                other_precedence == precedence
                and assoc == "R"
                and other_assoc is not None
            ):
                x = "({x})"

        if right_op:
            _, other_precedence, other_assoc = self.get_op_info(right_op)
            if other_precedence < precedence or (
                other_precedence == precedence
                and assoc == "L"
                and other_assoc is not None
            ):
                y = "({y})"

        return "%s %s %s" % (x, name, y)

    # noinspection PyUnusedLocal
    def transform_if_exp(self, test, body, orelse):
        return "xr.where({y}, {x}, {z})"

    def transform_bool_op(self, op, values):
        name, precedence, assoc = _ExprTranspiler.get_op_info(op)

        if name == "and" or name == "or":
            expr = None
            for i in range(1, len(values)):
                expr = "np.logical_%s(%s, {x%d})" % (
                    name,
                    "{x0}" if i == 1 else expr,
                    i,
                )
            return expr

        xes = []
        for i in range(len(values)):
            value = values[i]
            x = "{x%d}" % i
            other_op = getattr(value, "op", None)
            if other_op:
                _, other_precedence, other_assoc = self.get_op_info(other_op)
                if (
                    i == 0
                    and other_precedence < precedence
                    or i > 0
                    and other_precedence <= precedence
                ):
                    x = "(%s)" % x
            xes.append(x)

        return (" %s " % name).join(xes)

    # Compare(left, ops, comparators
    def transform_compare(self, left, ops, comparators):

        if len(ops) != 1:
            raise ValueError(
                'expression "%s" uses an n-ary comparison,'
                " but only binary are supported" % self.expr
            )

        right = comparators[0]
        op = ops[0]
        name, precedence, assoc = _ExprTranspiler.get_op_info(op)

        x = "{x0}"
        y = "{x1}"

        if self._is_nan(left):
            nan_op = y
        elif self._is_nan(right):
            nan_op = x
        else:
            nan_op = None

        if nan_op:
            if self.warn:
                warnings.warn(
                    "Use of NaN as operand with comparison"
                    ' "%s" is ambiguous: "%s"' % (name, self.expr)
                )
            if name == "==":
                return "np.isnan(%s)" % nan_op
            if name == "!=":
                return "np.logical_not(np.isnan(%s))" % nan_op

        left_op = getattr(left, "op", None)
        if left_op:
            name, other_precedence, assoc = _ExprTranspiler.get_op_info(left_op)
            if other_precedence < precedence:
                x = "(%s)" % x

        right_op = getattr(right, "op", None)
        if right_op:
            _, other_precedence, other_assoc = self.get_op_info(right_op)
            if other_precedence < precedence or (
                other_precedence == precedence
                and assoc == "L"
                and other_assoc is not None
            ):
                y = "(%s)" % y

        return "%s %s %s" % (x, name, y)

    def _is_nan(self, node):
        return isinstance(node, ast.Name) and node.id == "NaN"
