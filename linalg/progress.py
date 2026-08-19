"""Progress visualisation.

[Visualisable][linalg.progress.visualisable] functions display their progress
in different categories like scalar operations in the form of
[`tqdm`](https://tqdm.github.io/) progress bars.

When calling a [visualisable][linalg.progress.visualisable] function pass the
keyword argument `progress:Iterable[str]|bool` with the categories you want to
be visualised or `True`/`False` for all/none. For example

- `foo(..., progress=False, ...)`: (default) executes silently
- `foo(..., progress=True, ...)`: visualises all progresses
tracked by the function
- `foo(..., progress={'mul'}, ...)`: only visualises the progress of `mul`
operations

To make a function visualisable decorate it as
[`visualisable`][linalg.progress.visualisable].

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

The implementation now is more verbose, but behaves more predictably. This is
because someone writing a [`visualisable`][linalg.progress.visualisable]
function may not control where it gets hidden and called. Now it always knows
exactly if it is the top level callee, but it therefore has to always pass
progress downstream.
"""



from functools import wraps
from warnings import warn
from collections import Counter
from tqdm.auto import tqdm
from iteration import reduce_default, MISSING
from typing import Any
from collections.abc import Callable, Iterable, Sequence, Mapping



__all__ = ('Progress', 'visualisable')



class Progress:
    """Progress visualisation handler.
    
    Creates, updates and closes the progress bars.
    
    Use [`update`][linalg.progress.Progress.update] to increment the counts.
    
    Optionally use the scalar helpers, that basically behave like Pythons
    `operator` operators that also update.
    """
    
    announced: dict[str,int]
    counts: Counter
    pbars: dict[str,tqdm]
    
    def __init__(self, announced: Mapping[str,int],
                       selected: Iterable[str]|bool=False):
        self.announced = dict(announced)
        self.counts = Counter()
        if isinstance(selected, bool):
            selected = announced.keys() if selected else set()
        else:
            selected = set(selected)
            if not selected <= announced.keys():
                raise ValueError('untracked category requested')
        self.pbars = {k:tqdm(total=announced[k], desc=k) for k in selected}
    
    def update(self, category: str, n: int=1) -> None:
        """Increment progress bar for `category` by `n`, if it exists.
        
        Parameters
        ----------
        category : str
        n : int = 1
        """
        self.counts[category] += n
        if category in self.pbars:
            self.pbars[category].update(n)
    
    def close(self) -> None:
        """Close all progress bars."""
        for pbar in self.pbars.values():
            pbar.close()
    
    def _check(self) -> None:
        """Warn if any category didn't exactly update the announced amount."""
        mismatches = {
            k: (self.announced.get(k, 0), self.counts[k])
            for k in self.announced.keys() | self.counts.keys()
            if self.announced.get(k, 0) != self.counts[k]
        }
        if mismatches:
            warn(', '.join(f'{k!r} announced total {a} but {h} happened'
                           for k, (a, h) in mismatches.items()),
                 UserWarning, stacklevel=3)
    
    
    #scalar helpers
    def pos(self, a: Any) -> Any:
        """Return `+a`, and increment progress bar `pos`."""
        r = +a
        self.update('pos')
        return r
    
    def neg(self, a: Any) -> Any:
        """Return `-a`, and increment progress bar `neg`."""
        r = -a
        self.update('neg')
        return r
    
    def add(self, a: Any, b: Any) -> Any:
        """Return `a+b`, and increment progress bar `add`."""
        r = a + b
        self.update('add')
        return r
    
    def sub(self, a: Any, b: Any) -> Any:
        """Return `a-b`, and increment progress bar `sub`."""
        r = a - b
        self.update('sub')
        return r
    
    def mul(self, a: Any, b: Any) -> Any:
        """Return `a*b`, and increment progress bar `mul`."""
        r = a * b
        self.update('mul')
        return r
    
    def truediv(self, a: Any, b: Any) -> Any:
        """Return `a/b`, and increment progress bar `truediv`."""
        r = a / b
        self.update('truediv')
        return r
    
    def floordiv(self, a: Any, b: Any) -> Any:
        """Return `a//b`, and increment progress bar `floordiv`."""
        r = a // b
        self.update('floordiv')
        return r
    
    def mod(self, a: Any, b: Any) -> Any:
        """Return `a%b`, and increment progress bar `mod`."""
        r = a % b
        self.update('mod')
        return r
    
    def posneg(self, x: Any, sign: bool) -> Any:
        r"""Return `+x` if `bool(sign)==True` else `-x`.
        
        $$
            \begin{cases}
                +x & \text{if `sign`} \\
                -x & \text{otherwise}
            \end{cases}
        $$
        """
        return self.pos(x) if sign else self.neg(x)
    
    def sum_default(self, iterable: Iterable[Any], *,
            initial: Any=MISSING, default: Any=0) -> Any:
        """Return the sum of `iterable`.
        
        Without an unnecessary initial addition `+0`.
        
        Parameters
        ----------
        iterable : Iterable[Any]
        initial : Any = iteration.MISSING
        default: Any = 0
        
        Returns
        -------
        Any
            The sum.
        
        References
        ----------
        [`iteration.accumulators.sum_default`](https://goessl.github.io/iteration/accumulators/#iteration.accumulators.sum_default)
        """
        return reduce_default(self.add, iterable,
                initial=initial, default=default)
    
    def prod_default(self, iterable: Iterable[Any], *,
            initial: Any=MISSING, default: Any=1) -> Any:
        """Return the product of `iterable`.
        
        Without an unnecessary initial multiplication `*1`.
        
        Parameters
        ----------
        iterable : Iterable[Any]
        initial : Any = iteration.MISSING
        default : Any = 1
        
        Returns
        -------
        Any
            The product.
        
        References
        ----------
        [`iteration.accumulators.prod_default`](https://goessl.github.io/iteration/accumulators/#iteration.accumulators.prod_default)
        """
        return reduce_default(self.mul, iterable,
                initial=initial, default=default)
    
    def sumprod_default(self, a: Iterable[Any], b: Iterable[Any], *,
            initial: Any=MISSING, default: Any=0) -> Any:
        """Return the sum product of `a` & `b`.
        
        Without an unnecessary initial addition `+0`.
        
        Parameters
        ----------
        a, b : Iterable[Any]
        initial : Any = iteration.MISSING
        default : Any = 0
        
        Returns
        -------
        Any
            The sumproduct.
        
        References
        ----------
        [`iteration.accumulators.sumprod_default`](https://goessl.github.io/iteration/accumulators/#iteration.accumulators.sumprod_default)
        """
        return reduce_default(self.add, map(self.mul, a, b),
                initial=initial, default=default)



def visualisable(
    announcer: Callable[...,Mapping[str,int]],
    sanitiser: Callable[...,tuple[Sequence,Mapping]]|None=None):
    """Make the function visualisable.
    
    ```python
    @visualisable(f_announce, f_sanitise)
    def f(a, b, *, progress:Progress):
        ...
        z = x + y
        progress.update('add')
        ...
    ```
    
    where `f` is the executor, `f_announce` is the announcer
    and `f_sanitise` is the sanitiser.
    
    The optional sanitiser gets called first with the arguments excluding
    `progress:Iterable[str]|bool|Progress`. It shall do argument verification
    and preparation and return the operands as `tuple[Sequence, Mapping]`.
    For example you may want to accept [`numpy.typing.ArrayLike`](https://numpy.org/doc/stable/reference/typing.html#numpy.typing.ArrayLike)
    operands, that the sanitiser wraps in [`numpy.typing.NDArray`](https://numpy.org/doc/stable/reference/typing.html#numpy.typing.NDArray)s
    such that the announcer has access to shape information
    and the executor to indexing.
    
    The announcer receives the returnee of the sanitiser, or the raw arguments
    without `progress:Iterable[str]|bool|Progress` if there is no sanitiser.
    It shall return a `Mapping[str,int]` of tracked category names to their
    total number that will happen.
    
    The executor is the main decorated function. It receives the returnee of
    the sanitiser, or the raw arguments if there is no sanitiser, with an
    additional [`progress:Progress`][linalg.progress.Progress] object that
    handles the visualisation. It shall do the actual computation and update
    the progress handler via
    [`progress.update(category, n=1)`][linalg.progress.Progress.update].
    
    Parameters
    ----------
    announcer : Callable[...,Mapping[str,int]]
    sanitiser : Callable[...,tuple[Sequence,Mapping]]|None = None
    
    Returns
    -------
    Callable[[Callable],Callable]
        Decorator.
    
    Raises
    ------
    ValueError
        If visualisation of an untracked category is requested.
        This behaviour has been chosen to detect wrong usage.
    
    Warns
    -----
    UserWarning
        If the announced totals don't match the updates.
        Therefore always announce a hard upper limit and stuff the
        progress bars full on early exit.
        This behaviour has been chosen
        to continuously verify announcer implementations.
    
    Warnings
    --------
    Typing/signature annotations are currently terribly broken.
    
    The final wrapped function should have
    `(*args, progress:Iterable[str]|bool|Progress=False, **kwargs) -> ret`
    with `*args`, `**kwargs` from sanitiser and `ret` from the executor.
    The announcer and the executor should have the same receiving
    signature as the sanitiser return type. If no sanitiser is provided the
    resulting signature should be the one of the executor except
    `progress:Iterable[str]|bool` instead of `progress:Progress`.
    
    Current fix is to
    
    - disable mkdocstrings annotation rendering
    - document parameter types in docstring
    """
    def decorate(executor):
        @wraps(executor)
        def wrapper(*args, progress: Iterable[str]|bool|Progress=False,
                **kwargs):
            
            if sanitiser is not None:
                args, kwargs = sanitiser(*args, **kwargs)
            
            if owner := not isinstance(progress, Progress):
                progress = Progress(announcer(*args, **kwargs), progress)
            
            try:
                r = executor(*args, progress=progress, **kwargs)
                if owner:
                    progress._check()
                return r
            
            finally:
                if owner:
                    progress.close()
        
        return wrapper
    return decorate
