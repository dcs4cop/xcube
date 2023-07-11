# The MIT License (MIT)
# Copyright (c) 2023 by the xcube team and contributors
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
import inspect
from typing import Callable, Any, Dict, List

from .context import ComputeContext
from .op import get_operations


def get_compute_operations(ctx: ComputeContext):
    ops = get_operations()
    return {"operations": [encode_op(k, v) for k, v in ops.items()]}


def encode_op(op_id: str, op: Callable) -> Dict[str, Any]:
    members = dict(inspect.getmembers(op))
    print(members)
    return {
        "operationId": op_id,
        "parameters": []
    }


def encode_op_params(op_id: str, op: Callable) -> List[Dict[str, Any]]:
    return []


def encode_op_param(param_name: str, op: Callable) -> List[Dict[str, Any]]:
    return {}