from linalg import progress
import pytest
import warnings



#fully written by Claude



@pytest.fixture(autouse=True)
def announcements_must_be_right():
    """Turn every announcement mismatch into a test failure.

    `Progress` counts every key on every call, announced or not, so any
    visualisable call in the suite verifies its own announcer. Promoting the
    `UserWarning` here is what turns that from advice into a failure - kept
    out of `pyproject.toml` so the strictness belongs to the suite, not to
    anyone who happens to run `pytest` against an installed copy.
    `pytest.warns` installs its own filter, so tests that expect a warning
    are unaffected.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        yield



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
