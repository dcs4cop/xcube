import random
import time
import unittest
from typing import Sequence

import dask
import dask.array.random

from xcube.util.progress import ProgressObserver
from xcube.util.progress import ProgressState
from xcube.util.progress import add_progress_observers
from xcube.util.progress import new_progress_observers
from xcube.util.progress import observe_dask_progress
from xcube.util.progress import observe_nested_dask_progress
from xcube.util.progress import observe_progress


class MyProgressObserver(ProgressObserver):
    def __init__(self, record_errors=False):
        self.record_errors = record_errors
        self.calls = []

    def on_begin(self, state_stack: Sequence[ProgressState]):
        self.calls.append(('begin', self._serialize_stack(state_stack)))

    def on_update(self, state_stack: Sequence[ProgressState]):
        self.calls.append(('update', self._serialize_stack(state_stack)))

    def on_end(self, state_stack: Sequence[ProgressState]):
        self.calls.append(('end', self._serialize_stack(state_stack)))

    def _serialize_stack(self, state_stack):
        if self.record_errors:
            return [(s.label, s.progress, s.finished, s.exc_info_text) for s in state_stack]
        else:
            return [(s.label, s.progress, s.finished) for s in state_stack]


class ObserveProgressTest(unittest.TestCase):

    def test_observe_progress(self):
        observer = MyProgressObserver()
        observer.activate()

        with observe_progress('computing', 4) as reporter:
            # do something that takes 1 unit
            reporter.worked(1)
            # do something that takes 1 unit
            reporter.worked(1)
            # do something that takes 2 units
            reporter.worked(2)

        self.assertIsInstance(reporter.state, ProgressState)
        self.assertIsInstance(reporter.state.total_time, float)
        self.assertTrue(reporter.state.total_time >= 0.0)

        self.assertEqual(
            [
                ('begin', [('computing', 0.0, False)]),
                ('update', [('computing', 0.25, False)]),
                ('update', [('computing', 0.5, False)]),
                ('update', [('computing', 1.0, False)]),
                ('end', [('computing', 1.0, True)])
            ],
            observer.calls)

    def test_nested_observe_progress(self):
        observer = MyProgressObserver()
        observer.deactivate()
        observer.activate()

        with observe_progress('computing', 4) as reporter:
            # do something that takes 1 unit
            reporter.worked(1)
            # do something that takes 1 unit
            reporter.worked(1)
            # do something that will take 2 units
            reporter.will_work(2)
            with observe_progress('loading', 4) as nested_reporter:
                # do something that takes 3 units
                nested_reporter.worked(3)
                # do something that takes 1 unit
                nested_reporter.worked(1)

        self.assertEqual(
            [
                ('begin', [('computing', 0.0, False)]),
                ('update', [('computing', 0.25, False)]),
                ('update', [('computing', 0.5, False)]),
                ('begin', [('computing', 0.5, False), ('loading', 0.0, False)]),
                ('update', [('computing', 0.875, False), ('loading', 0.75, False)]),
                ('update', [('computing', 1.0, False), ('loading', 1.0, False)]),
                ('end', [('computing', 1.0, False), ('loading', 1.0, True)]),
                ('end', [('computing', 1.0, True)])
            ],
            observer.calls)

    def test_nested_observe_progress_with_new_progress_observers(self):
        observer = MyProgressObserver()
        observer.activate()

        nested_observer = MyProgressObserver()

        with observe_progress('computing', 4) as progress_reporter:
            # do something that takes 1 unit
            progress_reporter.worked(1)
            # do something that takes 1 unit
            progress_reporter.worked(1)
            with new_progress_observers(nested_observer):
                with observe_progress('loading', 4) as progress_reporter_2:
                    # do something that takes 3 units
                    progress_reporter_2.worked(3)
                    # do something that takes 1 unit
                    progress_reporter_2.worked(1)

            # do something that takes 1 unit
            progress_reporter.worked(2)

        self.assertEqual(
            [
                ('begin', [('computing', 0.0, False)]),
                ('update', [('computing', 0.25, False)]),
                ('update', [('computing', 0.5, False)]),
                ('update', [('computing', 1.0, False)]),
                ('end', [('computing', 1.0, True)])
            ],
            observer.calls)

        self.assertEqual(
            [
                ('begin', [('loading', 0.0, False)]),
                ('update', [('loading', 0.75, False)]),
                ('update', [('loading', 1.0, False)]),
                ('end', [('loading', 1.0, True)])
            ],
            nested_observer.calls)

    def test_nested_observe_progress_with_add_progress_observers(self):
        observer = MyProgressObserver()
        observer.activate()

        nested_observer = MyProgressObserver()

        with observe_progress('computing', 4) as reporter:
            # do something that takes 1 unit
            reporter.worked(1)
            # do something that takes 1 unit
            reporter.worked(1)
            # do something that will take 2 units
            reporter.will_work(2)
            with add_progress_observers(nested_observer):
                with observe_progress('loading', 4) as nested_reported:
                    # do something that takes 3 units
                    nested_reported.worked(3)
                    # do something that takes 1 unit
                    nested_reported.worked(1)

        self.assertEqual(
            [
                ('begin', [('computing', 0.0, False)]),
                ('update', [('computing', 0.25, False)]),
                ('update', [('computing', 0.5, False)]),
                ('begin', [('computing', 0.5, False), ('loading', 0.0, False)]),
                ('update', [('computing', 0.875, False), ('loading', 0.75, False)]),
                ('update', [('computing', 1.0, False), ('loading', 1.0, False)]),
                ('end', [('computing', 1.0, False), ('loading', 1.0, True)]),
                ('end', [('computing', 1.0, True)])
            ],
            observer.calls)

        self.assertEqual(
            [
                ('begin', [('computing', 0.5, False), ('loading', 0.0, False)]),
                ('update', [('computing', 0.875, False), ('loading', 0.75, False)]),
                ('update', [('computing', 1.0, False), ('loading', 1.0, False)]),
                ('end', [('computing', 1.0, False), ('loading', 1.0, True)])
            ],
            nested_observer.calls)

    def test_nested_observe_progress_with_exception(self):
        observer = MyProgressObserver(record_errors=True)
        observer.activate()

        try:
            with observe_progress('computing', 10) as reporter:
                # do something that takes 1 unit
                reporter.worked(1)
                # do something that takes 1 unit
                reporter.worked(1)
                # do something that will take 2 units
                reporter.will_work(8)
                with observe_progress('loading', 100) as nested_reported:
                    # do something that takes 3 units
                    nested_reported.worked(15)
                    # now - BANG!
                    raise ValueError('Failed to load')
        except ValueError:
            pass
        self.assertEqual(7, len(observer.calls))
        self.assertEqual(
            [
                ('begin', [('computing', 0.0, False, None)]),
                ('update', [('computing', 0.1, False, None)]),
                ('update', [('computing', 0.2, False, None)]),
                ('begin', [('computing', 0.2, False, None), ('loading', 0.0, False, None)]),
                ('update', [('computing', 0.32, False, None), ('loading', 0.15, False, None)]),
            ],
            observer.calls[0: -2])

        self.assertEqual(2, len(observer.calls[-2]))
        event, states = observer.calls[-2]
        self.assertEqual('end', event)
        self.assertEqual(2, len(states))
        self.assertEqual(4, len(states[0]))
        self.assertEqual(4, len(states[1]))
        self.assertEqual(('computing', 0.32, False), states[0][0:-1])
        self.assertEqual(('loading', 0.15, True), states[1][0:-1])
        error = states[0][-1]
        self.assertIsNone(error)
        error = states[1][-1]
        self.assertIsInstance(error, tuple)
        exc_type, exc_value, exc_traceback = error
        self.assertEqual('ValueError', exc_type)
        self.assertEqual('Failed to load', exc_value)
        self.assertIsInstance(exc_traceback, list)

        self.assertEqual(2, len(observer.calls[-1]))
        event, states = observer.calls[-1]
        self.assertEqual('end', event)
        self.assertEqual(1, len(states))
        self.assertEqual(4, len(states[0]))
        self.assertEqual(('computing', 0.32, True), states[0][0:-1])
        error = states[0][-1]
        self.assertIsInstance(error, tuple)
        exc_type, exc_value, exc_traceback = error
        self.assertEqual('ValueError', exc_type)
        self.assertEqual('Failed to load', exc_value)
        self.assertIsInstance(exc_traceback, list)

    def test_observe_asynchronous_progress(self):
        import asyncio

        async def do_something_async(i):
            with observe_progress(f"Doing something asynchronous #({i})", 3) as nested_reporter:
                await asyncio.sleep(0.01)
                nested_reporter.worked(1)
                await asyncio.sleep(0.01)
                nested_reporter.worked(1)
                await asyncio.sleep(0.01)
                nested_reporter.worked(1)

        async def gather_tasks():
            tasks = []
            for i in range(3):
                tasks.append(do_something_async(i + 1))
            await asyncio.gather(*tasks)

        observer = MyProgressObserver()
        observer.activate()
        with observe_progress("Reporting something asynchronous", 3) as reporter:
            reporter.will_work(3)
            asyncio.run(gather_tasks())

        self.assertEqual(('begin', [('Reporting something asynchronous', 0.0, False)]),
                         observer.calls[0])
        self.assertEqual(('end', [('Reporting something asynchronous', 3.0, True)]),
                         observer.calls[-1])


    def test_dask_progress(self):
        observer = MyProgressObserver(record_errors=True)
        observer.activate()

        res = dask.array.random.normal(size=(100, 200), chunks=(25, 50))
        with observe_dask_progress('computing', 100):
            res.compute()

        self.assertEqual(4, len(res.chunks[0]))
        self.assertTrue(len(observer.calls) >= 3)
        self.assertEqual(('begin', [('computing', 0.0, False, None)]), observer.calls[0])
        self.assertEqual(('update', [('computing', 5 / 16, False, None)]), observer.calls[5])
        self.assertEqual(('end', [('computing', 15 / 16, True, None)]), observer.calls[-1])


    def test_nested_dask_progress(self):
        observer = MyProgressObserver(record_errors=True)
        observer.activate()

        def _func(x):
            with observe_progress(f'Doing internal dask task {x}', 2) as nested_reporter:
                time.sleep(0.1 * random.random() * x)
                nested_reporter.worked(1)
                time.sleep(0.1 * random.random() * x)
                nested_reporter.worked(1)
                return x

        _func2 = dask.delayed(_func)
        x = _func2(1)
        y = _func2(2)
        z = _func2(3)
        def _nest_func(x, y, z):
            return x + y + z
        _nest_func2 = dask.delayed(_nest_func)
        a = _nest_func2(x, y, z)

        with observe_nested_dask_progress('Doing some tasks', 3):
            a.compute()

        self.assertTrue(len(observer.calls) == 14)
        self.assertEqual(('begin', [('Doing some tasks', 0.0, False, None)]), observer.calls[0])
        self.assertEqual(('end', [('Doing some tasks', 1.0, True, None)]), observer.calls[13])


class ProgressStateTest(unittest.TestCase):

    def test_progress_state_props(self):
        parent_state = ProgressState('observing', 3)
        parent_state.super_work_ahead = 3
        state = ProgressState('computing', 100, parent_state)
        self.assertEqual('computing', state.label)
        self.assertEqual(100, state.total_work)
        self.assertEqual(3, state.super_work)
        self.assertEqual(1, state.super_work_ahead)
        self.assertEqual(0, state.completed_work)
        self.assertEqual(0, state.progress)
        self.assertEqual(0.6, state.to_super_work(20))
        state.inc_work(15)
        self.assertEqual(15, state.completed_work)
        self.assertEqual(0.15, state.progress)
        self.assertEqual(0.6, state.to_super_work(20))
