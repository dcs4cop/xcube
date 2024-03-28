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

import functools
import importlib
import inspect
import concurrent.futures
import datetime
from typing import Dict, Any, Optional, Callable
import traceback

import xarray as xr

from xcube.core.mldataset import MultiLevelDataset
from xcube.server.api import Context, ApiError
from xcube.webapi.common.context import ResourcesContext
from xcube.webapi.datasets.context import DatasetsContext
from xcube.webapi.places import PlacesContext
from xcube.constants import LOG
from .op.info import OpInfo
from .op.registry import OpRegistry
from .op.registry import OP_REGISTRY

# Register default operations:
importlib.import_module("xcube.webapi.compute.operations")

LocalExecutor = concurrent.futures.ThreadPoolExecutor

# TODO: we should create a module 'job' and define better job classes.
#   Here we use dicts for time being.

Job = Dict[str, Any]
Jobs = Dict[int, Job]

JobRequest = Dict[str, Any]

JobFuture = concurrent.futures.Future
JobFutures = Dict[int, JobFuture]

JOB_INIT = "init"
JOB_SCHEDULED = "scheduled"
JOB_STARTED = "started"
JOB_COMPLETED = "completed"
JOB_FAILED = "failed"
JOB_CANCELLED = "cancelled"

JOB_STATUSES = {
    JOB_INIT,
    JOB_SCHEDULED,
    JOB_STARTED,
    JOB_COMPLETED,
    JOB_FAILED,
    JOB_CANCELLED,
}


class ComputeContext(ResourcesContext):
    """An xcube API server context for xcube compute operations

    Depends on the availability of `datasets` and `places` contexts
    in the same server context.
    """

    def __init__(self, server_ctx: Context, op_registry: OpRegistry = OP_REGISTRY):
        """Create a new compute context

        Args:
            server_ctx: the current server context object
            op_registry: the registry of compute operations to use for
                this context
        """
        super().__init__(server_ctx)
        self._datasets_ctx: DatasetsContext = server_ctx.get_api_ctx("datasets")
        assert isinstance(self._datasets_ctx, DatasetsContext)
        self._places_ctx: PlacesContext = server_ctx.get_api_ctx("places")
        assert isinstance(self._places_ctx, PlacesContext)

        self._op_registry = op_registry
        assert isinstance(self._op_registry, OpRegistry)

        compute_config = server_ctx.config.get("Compute", {})
        max_workers = compute_config.get("MaxWorkers", 3)

        self.next_job_id = 0
        self.jobs: Jobs = {}
        self.job_futures: JobFutures = {}
        self.job_executor = LocalExecutor(
            max_workers=max_workers, thread_name_prefix="xcube-job-"
        )

    def on_dispose(self):
        self.job_executor.shutdown(cancel_futures=True)

    @property
    def op_registry(self) -> OpRegistry:
        """
        Returns:
            the operation registry used by this compute context
        """
        return self._op_registry

    @property
    def datasets_ctx(self) -> DatasetsContext:
        """
        Returns:
            the datasets context used by this compute context
        """
        return self._datasets_ctx

    @property
    def places_ctx(self) -> PlacesContext:
        """
        Returns:
            the places context used by this compute context
        """
        return self._places_ctx

    def schedule_job(self, job_request: JobRequest) -> Job:
        """Schedule a new job given by *job_request*,
        which is expected to be validated already.

        Job status transitions:

        init --> pending --> cancelled
                         --> failed
                         --> running --> completed
                                     --> failed
                                     --> cancelled
        """

        with self.rlock:
            job_id = self.next_job_id
            self.next_job_id += 1

        # Note, the order of following statements is crucial:

        # Create new job
        job = new_job(job_id, job_request)
        # Register new job
        self.jobs[job_id] = job
        set_job_status(job, JOB_SCHEDULED)
        # Schedule job
        job_future: JobFuture = self.job_executor.submit(self._invoke_job, job_id)
        # Register new job future
        self.job_futures[job_id] = job_future
        # Notify when job is completed, failed, or cancelled.
        job_future.add_done_callback(functools.partial(self._handle_job_done, job_id))

        return job

    def _handle_job_done(self, job_id: int, job_future: JobFuture):
        job = self.jobs.get(job_id)
        if job is None:
            return

        if job_future.cancelled():
            set_job_status(job, JOB_CANCELLED)
        else:
            error = job_future.exception()
            if error:
                set_job_status(job, JOB_FAILED, error=error)
            else:
                set_job_status(job, JOB_COMPLETED)

        # Make sure we get rid of reference
        self.job_futures.pop(job_id, None)

    def _invoke_job(self, job_id: int):
        """After successfully scheduling a job this method call is
        represented by a Future in self.job_futures.

        Since it is executed concurrently, it cannot return
        anything to the original requester.
        For the same reason, raising an ApiError will not set the
        HTTP response status.
        """
        job = self.jobs.get(job_id)
        if job is None:
            return

        job_request = job["request"]
        op_id = job_request["operationId"]
        parameters = job_request.get("parameters", {})
        output = job_request.get("output", {})

        set_job_status(job, JOB_STARTED)

        op = self.op_registry.get_op(op_id)
        parameters = self.get_effective_parameters(op, parameters)

        # Execute the operation!
        output_ds = op(**parameters)

        ds_id = self.datasets_ctx.add_dataset(
            output_ds,
            ds_id=output.get("datasetId"),
            title=output.get("title"),
            style=output.get("style"),
            color_mappings=output.get("colorMappings"),
        )

        set_job_result(job, {"datasetId": ds_id})

    def cancel_job(self, job_id: int) -> Job:
        """Cancel a scheduled job.

        Args:
            job_id: the ID number of the job to be cancelled
        Returns:
            details of the cancelled job as a string-keyed dictionary
        Raises:
            ApiError: if the specified job cannot be found
        """
        job = self.jobs.get(job_id)
        if job is None:
            raise ApiError.NotFound(f"Job #{job_id} not found.")

        future = self.job_futures.pop(job_id, None)
        if future is not None and not future.done():
            set_job_status(job, JOB_CANCELLED)
            future.cancel()

        return job

    def get_effective_parameters(self, op: Callable, parameters: Dict[str, Any]):
        """Replace dataset names with datasets in operation parameters.

        This method takes a parameter dictionary for the invocation or an
        operation and returns a copy of the dictionary with any dataset
        names replaced by the actual referenced dataset. In other words,
        for any parameter where the operation expects a `Dataset` or
        `MultiLevelDataset` and the dictionary supplies a string, the string
        is replaced by the dataset with the corresponding name in this
        compute context’s dataset context.

        Args:
            op: an operation
            parameters: parameters with which to execute the operation
        Returns:
            a copy of the parameters, with dataset names replaced by
            datasets
        """
        op_info = OpInfo.get_op_info(op)
        param_py_types = op_info.effective_param_py_types
        parameters = parameters.copy()
        for param_name, param_py_type in param_py_types.items():
            param_value = parameters.get(param_name)
            if isinstance(param_value, str) and inspect.isclass(param_py_type):
                if issubclass(param_py_type, xr.Dataset):
                    parameters[param_name] = self._datasets_ctx.get_dataset(param_value)
                elif issubclass(param_py_type, MultiLevelDataset):
                    parameters[param_name] = self._datasets_ctx.get_ml_dataset(
                        param_value
                    )
        return parameters


def new_job(job_id: int, job_request: JobRequest) -> Job:
    return {
        "jobId": job_id,
        "request": job_request,
        "state": {"status": "init"},
    }


def is_job_status(job: Job, status: str) -> bool:
    """Report whether a specified job has the specified status.

    Args:
        job: a job specification (string-keyed dictionary)
        status: a string representing a recognized status

    Returns:
        True if the status is recognized and the specified job has the
        specified status

    Raises:
        ValueError: if the status is not recognized
    """
    _assert_valid_job_status(status)
    return job["state"]["status"] == status


def set_job_status(job: Job, status: str, error: Optional[BaseException] = None):
    """Set the status of a job.

    Args:
        job: a job specification (string-keyed dictionary)
        status: a string representing a recognized status
        error: if supplied, annotate the job specification with
            information from this exception

    Raises:
        ValueError: if the status is not recognized
    """
    _assert_valid_job_status(status)
    job_id = job["jobId"]
    LOG.info(f"Job #{job_id} {status}.", exc_info=error)
    # Note, we could/should warn/raise on invalid state transitions
    job["state"]["status"] = status
    job["state"][f"{status}Time"] = datetime.datetime.utcnow().isoformat()
    if error is not None:
        job["state"]["error"] = {
            "message": str(error),
            "type": type(error).__name__,
            "traceback": traceback.extract_tb(error.__traceback__).format(),
        }


def set_job_result(job: Job, result: Dict[str, Any]):
    """Set the result of a compute job.

    Args:
        job: job specifier (string-keyed dictionary)
        result: The result of the job
    """
    job["result"] = result


def _assert_valid_job_status(status):
    if status not in JOB_STATUSES:
        raise ValueError(f"illegal job status {status!r}")
