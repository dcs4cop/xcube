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

import threading
import time
import traceback
import weakref
from abc import ABC
from typing import Sequence, Optional, Any, Tuple, Type, List

import dask.callbacks

from xcube.util.assertions import assert_condition, assert_given, assert_in


class ProgressState:
    """Represents the state of progress."""

    def __init__(self, label: str, total_work: float,
                 parent_state: 'ProgressState' = None, trace: List[str] = None):
        self._label = label
        self._total_work = total_work
        self._ident = threading.get_ident()
        self._parent = parent_state
        self._stack = None
        self._super_work = parent_state.super_work_ahead if parent_state else 1
        self._super_work_ahead = 1.
        self._child_idents = []
        self._exc_info = None
        self._traceback = trace if trace else []
        self._completed_work = 0.
        self._finished = False
        self._start_time = None
        self._start_time = time.perf_counter()
        self._total_time = None

    @property
    def label(self) -> str:
        return self._label

    @property
    def total_work(self) -> float:
        return self._total_work

    @property
    def ident(self) -> int:
        return self._ident

    @property
    def parent(self) -> 'ProgressState':
        return self._parent

    @property
    def traceback(self) -> List[str]:
        return self._traceback

    @property
    def super_work(self) -> float:
        return self._super_work

    @property
    def completed_work(self) -> float:
        return self._completed_work

    @property
    def progress(self) -> float:
        return self._completed_work / self._total_work

    def to_super_work(self, work: float) -> float:
        return self._super_work * work / self._total_work

    @property
    def exc_info(self) -> Optional[Tuple[Type, BaseException, Any]]:
        return self._exc_info

    @exc_info.setter
    def exc_info(self, exc_info: Tuple[Type, BaseException, Any]):
        self._exc_info = exc_info

    @property
    def exc_info_text(self) -> Optional[Tuple[str, str, List[str]]]:
        if not self.exc_info:
            return None
        exc_type, exc_value, exc_traceback = self.exc_info
        return (f'{type(exc_value).__name__}',
                f'{exc_value}',
                traceback.format_exception(exc_type, exc_value, exc_traceback))

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def total_time(self) -> Optional[float]:
        return self._total_time

    @property
    def super_work_ahead(self) -> float:
        return self._super_work_ahead

    @super_work_ahead.setter
    def super_work_ahead(self, work: float):
        assert_condition(work > 0, 'work must be greater than zero')
        self._super_work_ahead = work

    def inc_work(self, work: float):
        assert_condition(work > 0, 'work must be greater than zero')
        self._completed_work += work
        if self._parent:
            work = self.to_super_work(work)
            self._parent.inc_work(work)

    def finish(self):
        self._finished = True
        self._total_time = time.perf_counter() - self._start_time

    @property
    def stack(self) -> List['ProgressState']:
        if not self._stack:
            self._stack = list()
            self._stack.append(self)
            previous_state = self._parent
            while previous_state:
                self._stack.insert(0, previous_state)
                previous_state = previous_state.parent
        return self._stack

    @property
    def child_idents(self) -> List[int]:
        return self._child_idents

    def set_child_idents(self, idents: List[str]):
        self._child_idents = idents


class ProgressObserver(ABC):
    """
    A progress observer is notified about nested state changes when using the
    :class:observe_progress context manager.
    """

    def on_begin(self, state_stack: Sequence[ProgressState]):
        """
        Called, if an observed code block begins execution.
        """

    def on_update(self, state_stack: Sequence[ProgressState]):
        """
        Called, if the progress state has changed within an observed code block.
        """

    def on_end(self, state_stack: Sequence[ProgressState]):
        """
        Called, if an observed block of code ends execution.
        """

    def activate(self):
        _ProgressContext.instance().add_observer(self)

    def deactivate(self):
        _ProgressContext.instance().remove_observer(self)


class _ProgressContext:
    _instance = None

    def __init__(self, *observers: ProgressObserver):
        self._observers = set(observers)
        self._states = set()

    def add_observer(self, observer: ProgressObserver):
        self._observers.add(observer)

    def remove_observer(self, observer: ProgressObserver):
        self._observers.discard(observer)

    def emit_begin(self, state_stack: List[ProgressState]):
        for observer in self._observers:
            observer.on_begin(state_stack)

    def emit_update(self, state_stack: List[ProgressState]):
        for observer in self._observers:
            observer.on_update(state_stack)

    def emit_end(self, state_stack: List[ProgressState]):
        for observer in self._observers:
            observer.on_end(state_stack)

    def begin(self, label: str, total_work: float) -> ProgressState:
        trace = traceback.format_stack()
        parent_state = self._get_parent_state(trace)
        progress_state = ProgressState(label, total_work, parent_state, trace)
        self._states.add(progress_state)
        self.emit_begin(progress_state.stack)
        return progress_state

    def _get_parent_state(self, traceback: List[str]) -> Optional[ProgressState]:
        parent_state = None
        max_match = 0
        ident = threading.get_ident()
        # preventing set modification during iteration
        states = self._states.copy()
        for state in states:
            if ident != state.ident:
                continue
            max_match_candidate = len(state.traceback)
            if len(traceback) < max_match_candidate or max_match_candidate < max_match:
                continue
            # the observer of the potential parent state is declared at index - 3 in the traceback
            # we check whether file and method match and the parent state's line number is smaller
            index = len(state.traceback) - 3
            if not self._may_be_parent_line(traceback[index], state.traceback[index]):
                continue
            index -= 1
            while index > 0:
                if traceback[index] != state.traceback[index]:
                    break
                index-=1
            if index > 0:
                continue
            parent_state = state
            max_match = max_match_candidate
        if parent_state is None:
            for state in self._states:
                if ident in state.child_idents:
                    parent_state = state
                    break
        return parent_state

    def _may_be_parent_line(self, state_traceback_line: str, parent_traceback_line: str):
        state_file_part, state_line_number, state_method_part = \
            self._dissect_traceline(state_traceback_line)
        parent_file_part, parent_line_number, parent_method_part = \
            self._dissect_traceline(parent_traceback_line)
        return state_file_part == parent_file_part and state_method_part == parent_method_part \
               and state_line_number > parent_line_number

    @staticmethod
    def _dissect_traceline(line: str):
        first_split_line = line.split('\n')[0]
        line_parts = first_split_line.split('line')
        file_part = line_parts[0]
        in_file_parts = line_parts[1].split(',')
        return file_part, int(in_file_parts[0]), in_file_parts[1]

    def end(self, progress_state, exc_type, exc_value, exc_traceback):
        exc_info = tuple((exc_type, exc_value, exc_traceback))
        progress_state.exc_info = exc_info if any(exc_info) else None
        progress_state.finish()
        self.emit_end(progress_state.stack)
        self._states.remove(progress_state)
        if progress_state.parent:
            progress_state.parent.super_work_ahead = 1

    def worked(self, progress_state: ProgressState, work: float):
        assert_in(progress_state, self._states)
        assert_condition(work > 0, 'work must be greater than zero')
        progress_state.inc_work(work)
        self.emit_update(progress_state.stack)

    def will_work(self, progress_state: ProgressState, work: float):
        assert_in(progress_state, self._states)
        progress_state.super_work_ahead = work

    @classmethod
    def instance(cls) -> '_ProgressContext':
        return cls._instance

    @classmethod
    def set_instance(cls, instance: '_ProgressContext' = None) -> '_ProgressContext':
        cls._instance, old_instance = (instance or _ProgressContext()), cls._instance
        return old_instance


_ProgressContext.set_instance()


class new_progress_observers:
    """
    Takes zero or more progress observers and activates them in the enclosed context.
    Progress observers from an outer context will no longer be active.

    :param observers: progress observers that will temporarily replace existing ones.
    """

    def __init__(self, *observers: ProgressObserver):
        self._observers = observers
        self._old_context = None

    def __enter__(self):
        self._old_context = _ProgressContext.set_instance(_ProgressContext(*self._observers))

    def __exit__(self, type, value, traceback):
        _ProgressContext.set_instance(self._old_context)


class add_progress_observers:
    """
    Takes zero or more progress observers and uses them only in the enclosed context.
    Any progress observers from an outer context remain active.

    :param observers: progress observers to be added temporarily.
    """

    def __init__(self, *observers: ProgressObserver):
        self._observers = observers

    def __enter__(self):
        for observer in self._observers:
            observer.activate()

    def __exit__(self, type, value, traceback):
        for observer in self._observers:
            observer.deactivate()


class observe_progress:
    """
    Context manager for observing progress in the enclosed context.

    :param label: A label.
    :param total_work: The total work.
    """

    def __init__(self, label: str, total_work: float):
        assert_given(label, 'label')
        assert_condition(total_work > 0, 'total_work must be greater than zero')
        self._label = label
        self._total_work = total_work
        self._state: Optional[ProgressState] = None

    @property
    def label(self) -> str:
        return self._label

    @property
    def total_work(self) -> float:
        return self._total_work

    @property
    def state(self) -> ProgressState:
        self._assert_used_correctly()
        return self._state

    def __enter__(self) -> 'observe_progress':
        self._state = _ProgressContext.instance().begin(self._label, self._total_work)
        return self

    def __exit__(self, type, value, traceback):
        _ProgressContext.instance().end(self._state, type, value, traceback)

    # noinspection PyMethodMayBeStatic
    def worked(self, work: float):
        self._assert_used_correctly()
        _ProgressContext.instance().worked(self._state, work)

    # noinspection PyMethodMayBeStatic
    def will_work(self, work: float):
        self._assert_used_correctly()
        _ProgressContext.instance().will_work(self._state, work)

    def _assert_used_correctly(self):
        assert_condition(self._state is not None,
                         'observe_progress() must be used with "with" statement')


class observe_dask_progress(dask.callbacks.Callback):
    """
    Observe progress made by Dask tasks.

    :param label: A label.
    :param total_work: The total work.
    :param interval: Time in seconds to between progress reports.
    :param initial_interval: Time in seconds to wait before progress is reported.
    """

    def __init__(self,
                 label: str,
                 total_work: float,
                 interval: float = 0.1,
                 initial_interval: float = 0):
        super().__init__()
        assert_given(label, 'label')
        assert_condition(total_work > 0, 'total_work must be greater than zero')
        self._label = label
        self._total_work = total_work
        self._state: Optional[ProgressState] = None
        self._initial_interval = initial_interval
        self._interval = interval
        self._last_worked = 0
        self._running = False

    def __enter__(self) -> 'observe_dask_progress':
        super().__enter__()
        self._state = _ProgressContext.instance().begin(self._label, self._total_work)
        return self

    def __exit__(self, type, value, traceback):
        self._stop_thread()
        _ProgressContext.instance().end(self._state, type, value, traceback)
        super().__exit__(type, value, traceback)

    # noinspection PyUnusedLocal
    def _start(self, dsk):
        """Dask callback implementation."""
        self._dask_state = None
        self._start_time = time.perf_counter()
        # Start background thread
        self._running = True
        self._timer = threading.Thread(target=self._timer_func)
        self._timer.daemon = True
        self._timer.start()

    # noinspection PyUnusedLocal
    def _pretask(self, key, dsk, state):
        """Dask callback implementation."""
        self._dask_state = state

    # noinspection PyUnusedLocal
    def _posttask(self, key, result, dsk, state, worker_id):
        """Dask callback implementation."""
        self._update()

    # noinspection PyUnusedLocal
    def _finish(self, dsk, state, errored):
        """Dask callback implementation."""
        self._stop_thread()
        elapsed = time.perf_counter() - self._start_time
        if elapsed > self._initial_interval:
            self._update()

    def _timer_func(self):
        """Background thread for updating"""
        while self._running:
            elapsed = time.perf_counter() - self._start_time
            if elapsed > self._initial_interval:
                self._update()
            time.sleep(self._interval)

    def _update(self):
        dask_state = self._dask_state
        if not dask_state:
            return
        num_done = len(dask_state['finished'])
        num_tasks = num_done + sum(len(dask_state[k]) for k in ['ready', 'waiting', 'running'])
        if num_done < num_tasks:
            work_fraction = num_done / num_tasks if num_tasks > 0 else 0
            worked = work_fraction * self._total_work
            work = worked - self._last_worked
            if work > 0:
                _ProgressContext.instance().worked(self._state, work)
                self._last_worked = worked

    def _stop_thread(self):
        if self._running:
            self._running = False
            self._timer.join()


class observe_nested_dask_progress(dask.callbacks.Callback):
    """
    Observe progress made by Dask tasks that have nested progress monitors.

    :param label: A label.
    :param num_measured_sub_processes: The number of sub processes with progress monitors.
    """

    def __init__(self,
                 label: str,
                 num_measured_sub_processes: float):
        super().__init__()
        assert_given(label, 'label')
        assert_condition(num_measured_sub_processes > 0, 'total_work must be greater than zero')
        self._label = label
        self._total_work = num_measured_sub_processes
        self._state: Optional[ProgressState] = None
        from multiprocessing.dummy import active_children
        self._active_children = active_children
        progress_thread = threading.current_thread()
        if not hasattr(progress_thread, '_children'):
            progress_thread._children = weakref.WeakKeyDictionary()

    def __enter__(self) -> 'observe_nested_dask_progress':
        super().__enter__()
        self._state = _ProgressContext.instance().begin(self._label, self._total_work)
        return self

    def __exit__(self, type, value, traceback):
        _ProgressContext.instance().end(self._state, type, value, traceback)
        super().__exit__(type, value, traceback)

    # noinspection PyUnusedLocal
    def _pretask(self, key, dsk, state):
        """Dask callback implementation."""
        child_idents = [active_child.ident for active_child in self._active_children()]
        self._state.set_child_idents(child_idents)
