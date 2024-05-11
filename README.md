# linalg

A linear algebra package for generic datatypes.
```python
>>> import numpy as np
>>> from fractions import Fraction
>>> from linalg import det_gauss
>>> 
>>> randf = lambda: Fraction(np.random.randint(-10, +11), np.random.randint(1, 10))
>>> A = np.array([[randf() for _ in range(5)] for _ in range(5)])
>>> det_gauss(A)
Fraction(3852719, 29160)
```

## Installation

```
pip install git+https://github.com/goessl/linalg.git
```

## Usage

This package provides the following functions:
- `det_gauss(A, criterion=bool)`: Determinant calculation by Gaussian elimination with complete pivoting.
- `inv_gauss(A, criterion=bool)`: Matrix inversion by Gaussian elimination with complete pivoting.
  - `swap_rows(A, i, j)`, `swap_columns(A, i, j)`, `swap_pivot(A, p, i, j)
- `row_echelon(M, reduced=True, criterion=bool)`: Transformation into (reduced) row echelon form.
  - `is_row_echelon(A, reduced=True, comparison=eq)`

## Conventions

- Matricies are represented as `numpy.ndarray`s.
- `criterion` argument: function that returns if the given element is non-zero.

## TODO

- [ ] progress visualisation
- [ ] readme

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
