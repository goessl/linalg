from linalg.progress import *
from linalg import progress
from linalg.blas import vadd
import numpy as np
import pytest
import warnings



#fully written by Claude
#the `bars` fixture lives in conftest.py, shared with test_blas2.py



@pytest.fixture
def counting():
    """Return a visualisable function performing exactly `n` additions."""
    @visualisable(lambda n: {'add': n})
    def f(n, *, progress:Progress):
        for _ in range(n):
            progress.update('add')

    return f


def miscounting(announced, performed):
    """Return a visualisable function announcing & performing different totals."""
    @visualisable(lambda: {'add': announced})
    def f(*, progress:Progress):
        for _ in range(performed):
            progress.update('add')

    return f



#selection
def test_naked_creates_nothing(bars):
    #without `progress` nothing is drawn at all
    vadd(np.arange(3), np.arange(3))
    assert bars.instances == []

def test_one_bar_per_requested_operation(bars):
    #the total is the broadcast size, not the shape
    vadd(np.arange(6).reshape(2, 3), np.arange(3), progress=('add',))
    assert [(b.desc, b.total) for b in bars.instances] == [('add', 6)]

def test_unrequested_operation_is_silent(bars):
    vadd(np.arange(3), np.arange(3), progress=('mul',))
    assert bars.instances == []

def test_unannounced_operation_is_silent(bars, counting):
    #`counting` announces only 'add', so a 'mul' request draws nothing
    counting(3, progress=('mul',))
    assert bars.instances == []



#counting
def test_counts_reach_the_total(bars):
    vadd(np.arange(6).reshape(2, 3), np.arange(3), progress=('add',))
    bar, = bars.instances
    assert bar.n == bar.total == 6

def test_update_of_unknown_operation_is_ignored(bars):
    @visualisable(lambda: {'add': 1})
    def f(*, progress:Progress):
        progress.update('add')
        progress.update('mul')      #no bar for this one

    f(progress=('add',))
    bar, = bars.instances
    assert bar.n == bar.total == 1



#ownership & nesting
def test_nested_call_does_not_open_a_second_bar(bars, counting):
    #the outermost function owns the display,
    #nested calls only count into its bars
    @visualisable(lambda n: {'add': 2*n})
    def twice(n, *, progress:Progress):
        counting(n, progress=progress)
        counting(n, progress=progress)

    twice(3, progress=('add',))
    bar, = bars.instances
    assert bar.n == bar.total == 6

def test_nested_call_does_not_close_the_bars(bars, counting):
    #only the owner closes, otherwise the outer bars die halfway
    @visualisable(lambda n: {'add': 2*n})
    def twice(n, *, progress:Progress):
        counting(n, progress=progress)
        assert not any(b.closed for b in bars.instances)
        counting(n, progress=progress)

    twice(3, progress=('add',))

def test_handed_a_handler_creates_nothing(bars, counting):
    #being passed a handler makes the callee a non-owner
    handler = Progress({'add': 3})
    bars.instances = []
    counting(3, progress=handler)
    assert bars.instances == []



#lifecycle
def test_bars_are_closed(bars, counting):
    counting(3, progress=('add',))
    assert all(b.closed for b in bars.instances)

def test_bars_are_closed_on_exception(bars):
    @visualisable(lambda: {'add': 3})
    def boom(*, progress:Progress):
        progress.update('add')
        raise ValueError

    with pytest.raises(ValueError):
        boom(progress=('add',))
    assert all(b.closed for b in bars.instances)



#announcement warnings
def test_matching_announcement_warns_nothing(bars, counting):
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        counting(3, progress=('add',))

def test_too_few_operations_warns(bars):
    with pytest.warns(UserWarning,
                      match='announced 5 operations but 3'):
        miscounting(5, 3)(progress=('add',))

def test_too_many_operations_warns(bars):
    with pytest.warns(UserWarning,
                      match='announced 3 operations but 5'):
        miscounting(3, 5)(progress=('add',))

def test_warning_names_the_operation(bars):
    with pytest.warns(UserWarning, match="'add'"):
        miscounting(2, 1)(progress=('add',))

def test_unrequested_operation_warns_nothing(bars):
    #no bar means there is nothing to compare against
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        miscounting(5, 3)(progress=('mul',))

def test_raising_call_warns_nothing(bars):
    #the bars are legitimately incomplete, the exception is the real message
    @visualisable(lambda: {'add': 3})
    def boom(*, progress:Progress):
        progress.update('add')
        raise ValueError

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with pytest.raises(ValueError):
            boom(progress=('add',))

def test_warning_points_at_the_calling_line(bars):
    #`stacklevel` must skip the wrapper, or the report is useless
    with pytest.warns(UserWarning) as record:
        miscounting(5, 3)(progress=('add',))
    assert record[0].filename == __file__

def test_nested_miscount_warns_once(bars, counting):
    #only the owner checks, so one warning per bar, not per nested call
    @visualisable(lambda n: {'add': 2*n})
    def twice(n, *, progress:Progress):
        counting(n, progress=progress)      #the second call is missing

    with pytest.warns(UserWarning) as record:
        twice(3, progress=('add',))
    assert len(record) == 1



#scalar helpers
@pytest.mark.parametrize('op, args, expected', [
    ('pos',       (3,),   +3),
    ('neg',       (3,),   -3),
    ('add',       (3, 2),  5),
    ('sub',       (3, 2),  1),
    ('mul',       (3, 2),  6),
    ('truediv',   (3, 2),  1.5),
    ('floordiv',  (3, 2),  1),
    ('mod',       (3, 2),  1),
])
def test_scalar_helper_returns_the_operation_result(op, args, expected):
    assert getattr(Progress({}), op)(*args) == expected

@pytest.mark.parametrize('op, args', [
    ('pos', (3,)),       ('neg', (3,)),
    ('add', (3, 2)),     ('sub', (3, 2)),       ('mul', (3, 2)),
    ('truediv', (3, 2)), ('floordiv', (3, 2)),  ('mod', (3, 2)),
])
def test_scalar_helper_increments_its_own_bar(bars, op, args):
    getattr(Progress({op: 1}), op)(*args)
    bar, = bars.instances
    assert (bar.desc, bar.n) == (op, 1)
