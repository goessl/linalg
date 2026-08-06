"""Progress visualisation.

Usage:

To visualise the progress of a visualisable function, pass the keyword argument
`progress:Iterable[str]` with the scalar operations you want to be visualised
(`add`, `sub`, ...).
The function will print its progress in the selected scalar operations as
[`tqdm`](https://tqdm.github.io/) progress bars.

Implementation:

To make your own functions progress visualisable, add the keyword parameter
`progress:Progress` to its signature and decorate it as
[`visualisable`][linalg.progress.visualisable].
[`visualisable`][linalg.progress.visualisable] needs a `Callable` that takes
the same parameters as the decorated function itself (excluding
`progress:Progress`) that at call time announces the total number of scalar
operations that will take place during execution.

The function then must either update its progress handler via
`progress.update(op:str, n:int=1)` or use the scalar helpers
`c = progress.add(a, b)`, and pass the handler down to subroutines.

If the number of announced operations doesn't equal the number of updates at
return a `UserWarning` is issued.

For example:

```python
>>> from itertools import zip_longest
... from time import sleep
... from linalg.progress import visualisable, Progress
... from typing import Any
... from collections.abc import Sequence
...
... @visualisable(lambda v, w: {'add':max(len(v), len(w))})
... def f(v:Sequence, w:Sequence, *, progress:Progress) -> list:
...     #Return vectorial `v + w`
...     r = []
...     for vi, wi in zip_longest(v, w, fillvalue=0):
...         r.append(vi + wi)
...         sleep(0.5)
...         progress.update('add')
...     return r
...
... @visualisable(lambda a, v, w: {'add':max(len(v), len(w)), 'mul':len(v)})
... def g(a: Any, v: Sequence, w: Sequence, *, progress:Progress) -> list:
...     #Return vectorial `a * v + w`
...     av = []
...     for vi in v:
...         sleep(0.5)
...         av.append(progress.mul(a, vi))
...     return f(av, w, progress=progress)
...
>>> v, w = [1, 2, 3], [4, 5, 6, 7]
>>> _ = f(v, w)
>>> _ = f(v, w, progress={'add'})
add: 100%|███████████████████████████████████| 4/4 [00:02<00:00,  2.00it/s]
>>> _ = g(2, v, w)
>>> _ = g(2, v, w, progress={'add'})
add: 100%|███████████████████████████████████| 4/4 [00:03<00:00,  1.14it/s]
>>> _ = g(2, v, w, progress={'add', 'mul'})
mul: 100%|███████████████████████████████████| 3/3 [00:03<00:00,  1.17s/it]
add: 100%|███████████████████████████████████| 4/4 [00:03<00:00,  1.14it/s]
```

Notes
-----
First a context & contextvars were used to trigger visualisation:
```python
with visualise('add', 'mul'):
    f(v, w)
    g(2, v, w)
```
but this led to surprising visualisations when the user called a custom
non-visualisable function that uses a visualisable function internally.

The implementation now is more verbose, but behaves more predictably.
"""



from functools import wraps
from warnings import warn
from tqdm.auto import tqdm
from typing import Any
from collections.abc import Iterable



__all__ = ('Progress', 'visualisable')



class Progress:
    """Progress visualisation handler.
    
    Creates, updates and closes the progress bars.
    
    Use [`update`][linalg.progress.Progress.update] to increment a progress bar.
    
    Optionally use the scalar helpers, that basically behave like Pythons
    `operator` operators, but additionally increment the corresponding progress
    bar.
    """
    
    pbars: dict[str, tqdm]
    
    def __init__(self, ops:dict[str,int]):
        self.pbars = {op:tqdm(total=n, desc=op) for op, n in ops.items()}
    
    def update(self, op:str, n:int=1) -> None:
        """Increment progress bar for operation `op` by `n`, if it exists."""
        if op in self.pbars:
            self.pbars[op].update(n)
    
    def close(self) -> None:
        """Close all progress bars."""
        for pbar in self.pbars.values():
            pbar.close()
    
    def _check(self) -> None:
        """Issue a warning if any progress bar isn't exactly filled to total."""
        for op, pbar in self.pbars.items():
            if pbar.n != pbar.total:
                warn(f'{op!r} announced {pbar.total} operations '
                     f'but {pbar.n} happened',
                     UserWarning, stacklevel=3)
    
    
    #scalar helpers
    def pos(self, a:Any) -> Any:
        """Return `+a`, and increment progress bar `pos`."""
        r = +a
        self.update('pos')
        return r
    
    def neg(self, a:Any) -> Any:
        """Return `-a`, and increment progress bar `neg`."""
        r = -a
        self.update('neg')
        return r
    
    def add(self, a:Any, b:Any) -> Any:
        """Return `a+b`, and increment progress bar `add`."""
        r = a + b
        self.update('add')
        return r
    
    def sub(self, a:Any, b:Any) -> Any:
        """Return `a-b`, and increment progress bar `sub`."""
        r = a - b
        self.update('sub')
        return r
    
    def mul(self, a:Any, b:Any) -> Any:
        """Return `a*b`, and increment progress bar `mul`."""
        r = a * b
        self.update('mul')
        return r
    
    def truediv(self, a:Any, b:Any) -> Any:
        """Return `a/b`, and increment progress bar `truediv`."""
        r = a / b
        self.update('truediv')
        return r
    
    def floordiv(self, a:Any, b:Any) -> Any:
        """Return `a//b`, and increment progress bar `floordiv`."""
        r = a // b
        self.update('floordiv')
        return r
    
    def mod(self, a:Any, b:Any) -> Any:
        """Return `a%b`, and increment progress bar `mod`."""
        r = a % b
        self.update('mod')
        return r



def visualisable(operations):
    """Make the function visualisable.
    
    ```python
    @visualisable(f_ops)
    def f(a, b, *, progress:Progress):
        ...
        z = x + y
        progress.update('add')
        ...
    ```
    
    Decorate a function that should be made visualisable.
    The decorator needs an additional function, that will receive the same
    arguments and should announce the total number of operations at call time
    the function will
    [`update` the `progress` handler][linalg.progress.Progress.update] during
    execution, including within subroutines.
    
    The number of operations that are announced
    at call time must match the total of updates during execution.
    
    Notes
    -----
    Typing not yet achieved perfectly without deviating from a clean logic
    implementation. Therefore untyped.
    """
    def decorate(function):
        @wraps(function)
        def wrapper(*args, progress:Iterable[str]|Progress=(), **kwargs):
            
            if owner := not isinstance(progress, Progress):
                ops = operations(*args, **kwargs)
                ops = {op:ops[op] for op in progress if op in ops}
                progress = Progress(ops)
            
            try:
                r = function(*args, progress=progress, **kwargs)
                if owner:
                    progress._check()
                return r
            
            finally:
                if owner:
                    progress.close()
        
        return wrapper
    return decorate
