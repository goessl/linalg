from linalg.progress import *
from linalg import progress
from linalg.blas import vadd
import numpy as np
import pytest



#fully written by Claude



#supervising classes
class FakeBar:
    """Stand-in for `tqdm` that records instead of renders.
    
    Replaces the `tqdm` name in `linalg.progress`, which is used both to
    construct bars and to write labels, so one class covers both.
    The rendered output itself is not asserted on: it carries rates, elapsed
    times and terminal-width dependent padding, and changes between `tqdm`
    versions.
    """
    instances: list['FakeBar'] = []
    written: list[str] = []
    
    def __init__(self, total=None, desc=None, position=None, **kwargs):
        self.total, self.desc, self.position = total, desc, position
        self.n, self.closed = 0, False
        FakeBar.instances.append(self)
    
    def update(self, amount=1):
        self.n += amount
    
    def close(self):
        self.closed = True
    
    @staticmethod
    def write(text):
        FakeBar.written.append(text)


@pytest.fixture
def bars(monkeypatch):
    """Return the `FakeBar` that `linalg.progress` draws with, freshly reset."""
    FakeBar.instances, FakeBar.written = [], []
    monkeypatch.setattr(progress, 'tqdm', FakeBar)
    return FakeBar



#tests
def test_naked_creates_nothing(bars):
    #outside `visualise` nothing is drawn at all
    vadd(np.arange(3), np.arange(3))
    assert bars.instances == [] and bars.written == []

def test_one_bar_per_requested_operation(bars):
    #the total is the broadcast size, not the shape
    with visualise('add'):
        vadd(np.arange(6).reshape(2, 3), np.arange(3))
    assert bars.written == ['vadd']
    assert [(b.desc, b.total) for b in bars.instances] == [('add', 6)]

def test_counts_reach_the_total(bars):
    with visualise('add'):
        vadd(np.arange(6).reshape(2, 3), np.arange(3))
    bar, = bars.instances
    assert bar.n == bar.total == 6

def test_unrequested_operation_is_silent(bars):
    with visualise('mul'):
        vadd(np.arange(3), np.arange(3))
    assert bars.instances == [] and bars.written == []

def test_bars_are_closed(bars):
    with visualise('add'):
        vadd(np.arange(3), np.arange(3))
    assert all(b.closed for b in bars.instances)

def test_bars_are_closed_on_exception(bars):
    @visualisable(lambda a: {'add': np.broadcast(a).size})
    def boom(a):
        notify('add')
        raise ValueError
    
    with pytest.raises(ValueError), visualise('add'):
        boom(np.arange(3))
    assert all(b.closed for b in bars.instances)

def test_nested_call_does_not_open_a_second_bar(bars):
    #the outermost visualisable function owns the display,
    #the nested `vadd`s only count into its bar
    @visualisable(lambda a: {'add': 2*np.broadcast(a).size})
    def twice(a):
        vadd(a, a)
        vadd(a, a)
    
    with visualise('add'):
        twice(np.arange(3))
    assert bars.written == ['twice']
    assert len(bars.instances) == 1
    assert bars.instances[0].n == 6

def test_under_declaring_parent_silences_children(bars):
    #`h` claims to do no additions, so its children stay quiet
    #instead of each opening their own bar
    @visualisable(lambda a: {'mul': np.broadcast(a).size})
    def h(a):
        vadd(a, a)
    
    with visualise('add'):
        h(np.arange(3))
    assert bars.instances == [] and bars.written == []
