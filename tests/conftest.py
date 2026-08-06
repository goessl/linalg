from linalg import progress
import pytest



#fully written by Claude



class FakeBar:
    """Stand-in for `tqdm` that records instead of renders.

    Replaces the `tqdm` name in `linalg.progress`, which is where every bar
    is constructed.
    The rendered output itself is not asserted on: it carries rates, elapsed
    times and terminal-width dependent padding, and changes between `tqdm`
    versions.
    """
    instances: list['FakeBar'] = []

    def __init__(self, total=None, desc=None, **kwargs):
        self.total, self.desc = total, desc
        self.n, self.closed = 0, False
        FakeBar.instances.append(self)

    def update(self, amount=1):
        self.n += amount

    def close(self):
        self.closed = True


@pytest.fixture
def bars(monkeypatch):
    """Return the `FakeBar` that `linalg.progress` draws with, freshly reset."""
    FakeBar.instances = []
    monkeypatch.setattr(progress, 'tqdm', FakeBar)
    return FakeBar
