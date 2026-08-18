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

def test_one_bar_per_requested_category(bars):
    #the total is the broadcast size, not the shape
    vadd(np.arange(6).reshape(2, 3), np.arange(3), progress=('add',))
    assert [(b.desc, b.total) for b in bars.instances] == [('add', 6)]

def test_untracked_category_raises(bars):
    #`vadd` announces only 'add', so 'mul' is a user mistake, not a silence
    with pytest.raises(ValueError, match='untracked'):
        vadd(np.arange(3), np.arange(3), progress=('mul',))

def test_untracked_category_raises_before_drawing(bars):
    with pytest.raises(ValueError):
        vadd(np.arange(3), np.arange(3), progress=('mul',))
    assert bars.instances == []

def test_one_untracked_category_rejects_the_whole_request(bars, counting):
    #'add' alone would be fine, the 'mul' next to it is what fails
    with pytest.raises(ValueError, match='untracked'):
        counting(3, progress=('add', 'mul'))

def test_progress_true_needs_no_category_names(bars, counting):
    #the bool shortcut can not name an untracked category
    counting(3, progress=True)
    bar, = bars.instances
    assert (bar.desc, bar.n, bar.total) == ('add', 3, 3)



#counting
def test_update_of_unannounced_category_warns(bars):
    #counting is always on, so a category the announcer never declared is caught
    #even though it is not rendered
    @visualisable(lambda: {'add': 1})
    def f(*, progress:Progress):
        progress.update('add')
        progress.update('mul')      #never announced

    with pytest.warns(UserWarning, match="'mul' announced total 0 but 1"):
        f(progress=('add',))



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



#sanitiser
@pytest.fixture
def sanitising():
    """Return a visualisable function whose sanitiser doubles its argument."""
    @visualisable(lambda n: {'add': n}, lambda n: ([2*n], {}))
    def f(n, *, progress:Progress):
        for _ in range(n):
            progress.update('add')
        return n

    return f

def test_sanitiser_normalises_the_arguments(bars, sanitising):
    assert sanitising(3) == 6

def test_sanitiser_runs_before_the_announcer(bars, sanitising):
    #the announcer must see the sanitised argument, not the raw one
    sanitising(3, progress=('add',))
    bar, = bars.instances
    assert bar.total == 6

def test_sanitiser_runs_for_a_non_owner_too(bars, sanitising):
    #a nested call arrives with a handler & skips the ownership branch,
    #so the sanitiser must sit outside it or the callee gets raw arguments
    assert sanitising(3, progress=Progress({})) == 6

def test_sanitiser_rejects_before_any_bar_is_drawn(bars):
    @visualisable(lambda n: {'add': n}, lambda n: 1/0)
    def f(n, *, progress:Progress):
        pass

    with pytest.raises(ZeroDivisionError):
        f(3, progress=('add',))
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
def test_too_few_updates_warns(bars):
    with pytest.warns(UserWarning,
                      match='announced total 5 but 3'):
        miscounting(5, 3)(progress=('add',))

def test_too_many_updates_warns(bars):
    with pytest.warns(UserWarning,
                      match='announced total 3 but 5'):
        miscounting(3, 5)(progress=('add',))

def test_warning_names_the_category(bars):
    with pytest.warns(UserWarning, match="'add'"):
        miscounting(2, 1)(progress=('add',))

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
    getattr(Progress({op: 1}, True), op)(*args)
    bar, = bars.instances
    assert (bar.desc, bar.n) == (op, 1)
