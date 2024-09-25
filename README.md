# linalg

A type-independent and exact linear algebra package.
```python
>>> import numpy as np
>>> from fractions import Fraction
>>> from linalg import det_laplace
>>> A = np.array([[Fraction( 9, 5), Fraction(-3, 4), Fraction( -1, 4)],
...               [Fraction( 5, 7), Fraction(-7, 2), Fraction(  9, 4)],
...               [Fraction(10, 9), Fraction( 1, 7), Fraction(-10, 7)]])
>>> det_laplace(A)
Fraction(84379, 17640)
```

## Installation

```console
pip install git+https://github.com/goessl/linalg.git
```

## Usage

### Leibniz

- `det_leibniz(A)`: Return the determinant of `A`. Calculates the determinant by the Leibniz formula.
- `minor_leibniz(A, i, j)`: Return the `i,j`-th minor of `A`. See `det_leibniz` for more.

### Laplace

- `det_laplace(A)`: Return the determinant of `A`. Calculates the determinant by Laplace expansion. Uses the row/column with the most zero elements.
- `minor_laplace(A, i, j)`: Return the `i,j`-th minor of `A`. See `det_laplace` for more.

### Bareiss

- `det_bareiss(A)`: Return the determinant of an integer matrix `A`. Calculates the determinant by the Bareiss algorithm. Transforms `A` in place.

### Random Creation

- `randint(low, high=None, size=None)`: Return a random `int` or an array of random `int`s. Like `numpy.random.randint` but with native `int`s.
- `randf(shape=None, precision=1000)`: Return a random `Fraction` or an array of random `Fraction`s. Their numerators `n` are `-precision <= n <= +precision` and their denominators `d` are `1 <= d <= +precision`.

### Utility

- `_prod(iterable)`: Like `math.prod` but for non-numeric types. `math.prod` might reject non-numeric types: https://docs.python.org/3/library/math.html#math.prod. For `float`s keep using `math.prod` for better precision.
- `swap_rows(A, i, j)`: Swap the `i`-th and `j`-th row of `A` in-place.
- `swap_columns(A, i, j)`: Swap the `i`-th and `j`-th column of `A` in-place.
- `swap_pivot(A, p, i, j)`: Swap the `p`-&`i`-th rows and `p`-&`j`-th columns of `A` in-place.
- `submatrix(A, i, j)`: Return a copy of `A` without the `i`-th row and `j`-th column.
- `_permutations(iterable, r=None)`: `itertools.permutation`, but yields `permutation, parity`.

## Conventions

- Matrices are represented as `numpy.ndarray`s.

## todo

- [ ] `sum` & `prod` for non-numeric types (`math.prod` might reject non-numerics; both have `start` argument with unnecessary additional operation; but have better float precision -> automatic fallback?)
- [ ] complexities
- [ ] progress visualisation with `tqdm`
- [ ] minimum dimensions (do functions work for 0x0-matrices?)

## License (MIT)

Copyright (c) 2024 Sebastian Gössl

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
