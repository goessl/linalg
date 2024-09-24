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

### Laplace

- `det_laplace(A)`: Return the determinant of `A`. Calculates the determinant by Laplace expansion. Uses the row/column with the most zero elements.
- `minor_laplace(A, i, j)`: Return the `i,j`-th minor of `A`. See `det_laplace` for more.

### Utility

- `randf(shape=None, precision=1000)`: Return a random `Fraction` or an array of random `Fraction`s. Their numerators `n` are `-precision <= n <= +precision` and their denominators `d` are `1 <= d <= +precision`.
- `submatrix(A, i, j)`: Return a copy of `A` without the `i`-th row and `j`-th column.

## todo

 - [ ] complexities
 - [ ] `tqdm`

## License (MIT)

Copyright (c) 2022-2024 Sebastian Gössl

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
