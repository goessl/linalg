# linalg

A type-independent and exact linear algebra package with progress visualisation.
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

## Key features

- *Generic Object Support*: Perform linear algebra operations on matrices with elements like Python’s `fractions.Fraction`. This allows exact computations, unlike other libraries like NumPy or SciPy, which mainly support floating-point operations.
- *Operation Tracking*: Every function in this package provides an exact count of the arithmetic operations performed, helping you choose the best algorithm for your custom data types.
- *Progress Visualization*: Coming soon! You'll be able to track the progress of long-running calculations, particularly useful when using slow custom data types.

## Installation

```console
pip install git+https://github.com/goessl/linalg.git
```

## Usage

### `linalg.matmul`

- `matmul(A, B, progress=set())`: Return the matrix product of `A` & `B`.
  Matrices must be non-empty (L,N,M>0).
  For a LxN & a NxM matrix (where the result will be LxM) there will be
  - $L(N-1)M$ additions (`+`),
  - $LNM$ multiplications (`*`),
  - so $L(2N-1)M$ arithmetic operations in total.

### `linalg.backsub`

- `backsub(T, y, tril=False)`: Return the solution for `Tx=y` where the matrix `T` is triangular.
  Back substitution like BLAS `xTRSV`.
  `tril=False` if `T` is upper triangular, `tril=True` if `T` is lower triangular.
  For a N vector/NxN matrix there will be
  - $N(N-3)/2+1$ additions (`+`),
  - $N-1$ subtractions (`-`),
  - $N(N-1)/2$ multiplications (`*`),
  - $N$ divisions (`/`),
  - so $N^2$ arithmetic operations in total.

### `linalg.leibniz`

- `det_leibniz(A)`: Return the determinant of `A`.
  Calculates the determinant by the Leibniz formula.
  For a NxN matrix there will be at most
  - $N!-1$ additions (`+`),
  - $(N-1)N!$ multiplications (`*`),
  - so $NN!-1$ arithmetic operations in total.

### `linalg.laplace`

- `det_laplace(A)`: Return the determinant of `A`.
  Calculates the determinant by Laplace expansion.
  Uses the row/column with the most zero elements.
  For a NxN matrix there will be at most
  - $N!-1$ additions (`+`) and subtractions (`-`),
  - $floor((e-1)N!)-1$ multiplications (`*`) (https://oeis.org/A038156),
  - so $floor(eN!)-2$ arithmetic operations in total (https://oeis.org/A026243).

### `linalg.gauss`

- `det_gauss(A)`: Return the determinant of `A`.
  Calculates the determinant by Gaussian elimination with complete pivoting.
  The matrix will be transformed in place into an upper triangular matrix (columns left of pivot won't be reduced).
  For a NxN matrix there will be
  - $N(N^2-1)/3$ subtractions (`-`),
  - $N(N^2+2)/3-1$ multiplications (`*`),
  - $N(N-1)/2$ divisions (`/`),
  - so $N(4N^2+3N-1)/6-1$ arithmetic operations in total.
- `inv_gauss(A)`: Return the inverse of `A`.
  Calculates the inverse by Gaussian elimination with complete pivoting.
  The matrix will be transformed in place into the identity matrix.
  For a NxN matrix there will be
  - $N^3-N^2$ additions (`+`),
  - $2N^3-3N^2+2N-1$ subtractions (`-`),
  - $2N^3-2N^2$ multiplications (`*`),
  - $2N^2-N+1$ divisions (`/`),
  - so $5N^3-4N^2+N$ arithmetic operations in total.

### `linalg.bareiss`

- `det_bareiss(A)`: Return the determinant of an integer matrix `A`.
  Calculates the determinant by the Bareiss algorithm.
  Transforms `A` in place.
  For a NxN matrix there will be
  - $N(N^2-1)/3$ subtractions (`-`),
  - $2N(N^2-1)/3$ multiplications (`*`),
  - $N(N^2-3N+2)/3$ floor divisions (`//`),
  - so $N(4N^2-3N-1)/3$ arithmetic operations in total.

### `linalg.adjugate`

- `minor(A, i, j, det=det_gauss)`: Return the `i,j`-th minor of `A`.
  Uses the given determinant algorithm.
- `cofactor(A, i, j, det=det_gauss)`: Return the `i,j`-th cofactor of `A`.
  Uses the given determinant algorithm.
- `adj(A, det=det_gauss)`: Return the adjugate of `A`.
  Uses the given determinant algorithm.
- `cof(A, det=det_gauss)`: Return the cofactor matrix of `A`.
  Uses the given determinant algorithm.

### `linalg.lu`

- `is_perm(P)`: Return if `P` is a permutation matrix.
- `is_tril(L)`: Return if `L` is lower triangular.
- `is_triu(U)`: Return if `U` is upper triangular.
- `LU(A)`: Return the LU decomposition of `A`.
  Transformation happens in-place (`A` becomes `U` for Doolittle, `L` for Crout).
  For a MxN matrix returns a MxM- & MxN-matrix for Doolittle or a MxN & NxN matrix for Crout (automatically most memory efficient).
  Doolittle decomposition ($l_{ii}=1$) if `M<=N`, Crout decomposition ($u_{ii}=1$) otherwise.
  https://en.wikipedia.org/wiki/LU_decomposition#LU_Crout_decomposition
  For a matrix (where `M` is the shorter and `N` the longer side length) there will be
  - $M(M-1)N/2$ subtractions (`-`),
  - $M(M-1)N/2$ multiplications (`*`),
  - $M(M-1)/2$ divisions (`/`),
  - so $M(M-1)N+M(M-1)/2$ arithmetic operations in total.
- `PLU(A)`: Return the PLU decomposition of `A`.
  Transformation happens in-place (`A` becomes `U`).
  For a MxN matrix returns a MxM, MxM & MxN matrix.
  Doolittle decomposition ($L_{ii}=1$).
  https://en.wikipedia.org/wiki/LU_decomposition#LU_Crout_decomposition
  If `M>N` `LUQ` would be more efficient.
  For a MxN matrix (`O=min(M-1, N)`) there will be
  - $NO(2M-O-1)/2$ subtractions (`-`),
  - $NO(2M-O-1)/2$ multiplications (`*`),
  - $O(2M-O-1)/2$ divisions (`/`),
  - so $NO(2M-O-1)+O(2M-O-1)/2$ arithmetic operations in total.
- `LUQ(A)`: Return the LUQ decomposition of `A`.
  Transformation happens in-place (`A` becomes `L`).
  For a MxN matrix returns a MxN, NxN & NxN matrix.
  Crout decomposition ($U_{ii}=1$).
  https://en.wikipedia.org/wiki/LU_decomposition#LU_Crout_decomposition
  If `M<N` `PLU` would be more memory efficient.
  For a MxN matrix (`O=min(M, N-1)`) there will be
  - $MO(2N-O-1)/2$ subtractions (`-`),
  - $MO(2N-O-1)/2$ multiplications (`*`),
  - $O(2N-O-1)/2$ divisions (`/`),
  - so $MO(2N-O-1)+O(2N-O-1)/2$ arithmetic operations in total.
- `PLUQ(A)`: Return the PLUQ decomposition of `A`.
  Transformation happens in-place (`A` becomes `U` for Doolittle, `L` for Crout).
  For a MxN matrix returns a MxM, MxM, MxN & NxN matrix for Doolittle or a MxM, MxN, NxN & NxN matrix for Crout (automatically most memory efficient).
  Doolittle decomposition ($l_{ii}=1$) if `M<=N`, Crout decomposition ($u_{ii}=1$) otherwise.
  https://en.wikipedia.org/wiki/LU_decomposition#LU_Crout_decomposition
  For a matrix (where `M` is the shorter and `N` the longer side length) there will be
  - $M(M-1)N/2$ subtractions (`-`),
  - $M(M-1)N/2$ multiplications (`*`),
  - $M(M-1)/2$ divisions (`/`),
  - so $M(M-1)N+M(M-1)/2$ arithmetic operations in total.

### `linalg.rank`

- `is_ref(A, reduced=True)`: Return if `A` is of (reduced) row echelon form.
- `ref_gauss(A, reduced=True)`: Transform `A` into (reduced) row echelon form.
  By Gaussian elimination with pivoting.
  Transforms `A` in place.
  For a MxN-matrix of rank r there will be
  - $MNr-Nr(r+1)/2$ subtractions (`-`),
  - $MNr-Nr(r+1)/2$ multiplications (`*`),
  - $Mr-r(r+1)/2$ divisions (`/`),
  - so $Mr(2N+1)-Nr(r+1)-r(r+1)/2$ operations in total.
- `rank_decomposition(A)`: Return a rank decomposition `B, C` of `A` such that `A=BC`.
- `pinv(A)`: Return the Moore–Penrose inverse of `A`.
- `lstsq(X, y)`: Return the linear least squares solution `b` for `y=Xb`.

### `linalg.rand`

- `randint(low, high=None, size=None)`: Return a random `int` or an array of random `int`s.
  Like `numpy.random.randint` but with native `int`s.
- `randf(shape=None, precision=1000)`: Return a random `Fraction` or an array of random `Fraction`s.
  Their numerators `n` are `-precision <= n <= +precision` and their denominators `d` are `1 <= d <= +precision`.
- `randfr(M, N, r, precision=1000)`: Return a random MxN `Fraction` matrix of rank `r`.

### `linalg.util`

- `posneg(x, s)`: Return `+x` if `s==True` or `-x` if `s==False`.
  To use the unary operators `+` (`__pos__`) & `-` (`__neg__`) instead of multiplication with `+1` & `-1` (`__mul__`) as the univariate operators may be more optimised and multiplication with integers doesn't have to be implemented (might be the case for custom prototyped number types).
- `_sum(iterable)`: Like `sum` but without initial `+0`.
  For `float`s keep using `sum` for better precision.
  See [README.md#todo](README.md#Conventions) for justification.
- `_prod(iterable)`: Like `math.prod` but for non-numeric types.
  `math.prod` might reject non-numeric types: https://docs.python.org/3/library/math.html#math.prod.
  For `float`s keep using `math.prod` for better precision.
  See [README.md#todo](README.md#Conventions) for justification.
- `assert_matrix(A)`: Assert matrix.
- `assert_sqmatrix(A)`: Assert square matrix.
- `swap_rows(A, i, j)`: Swap the `i`-th and `j`-th row of `A` in-place.
- `swap_columns(A, i, j)`: Swap the `i`-th and `j`-th column of `A` in-place.
- `swap_pivot(A, p, i, j)`: Swap the `p`-&`i`-th rows and `p`-&`j`-th columns of `A` in-place.
- `submatrix(A, i, j)`: Return a copy of `A` without the `i`-th row and `j`-th column.
- `_permutations(iterable, r=None)`: `itertools.permutation`, but yields `permutation, parity`.

### `linalg.progress`

- `Progress`: Progress handler for algorithms.
  Use as context.
  Total number of operations must be known beforehand.
  Call `update(op, n=1)` to increment tracking.
  - `__init__(totals, descprefix='')`: Create a new progress handler.
    `totals` should be a dictionary with the tracked operations as keys and the total number of operations as values.
  - `__enter__()`
  - `__exit__(type, value, traceback)`
  - `update(op, n=1)`: Increment the operation `op` progress by `n`.
    If `op` is not tracked nothing happens.

### `linalg.counterwrapper`

- `CounterWrapper`
- `toCounterWrappers`
- `fromCounterWrappers`

## Conventions

- Matrices are represented as `numpy.ndarray`s.
- Elements are assumed to be exact. E.g. Gaussian elemination only stops if the pivot `A[i, i]` results in `bool(A[i, i])==False`. Or it doesn't reduce elements to the lower left of the pivot because they should already be eliminated to zero.
- Elements obviously must support the required scalar operations. Arithmetic operations are mentioned in the complexity analysis of the respective docstring. More may be needed like boolean evaluation and comparisons.
- Complexity analysis only cares about elemental binary arithmetic operations. Assignments, boolean evaluation, comparisons, unitary `+` & `-`, ... are not counted (this can shift subtractions to additions because `a-b=a+(-b)`, and therefore are additions and subtractions counted together in functions that use `posneg` in sums).
- Arithmetic operations with special integers (`+-int(0)`, `*int(+-1)`, `/int(1)`, `int(0)/`) are avoided whenever possible for faster code and correct complexity analysis (some custom number types may be badly optimised/would need branching in arithmetic methods, e.g. [goessl/sqrtfractions](https://github.com/goessl/sqrtfractions)). This is done with utility functions like `posneg`, `_sum` & `_prod`.
- Progress visualisation is done with the `progress=set()` argument. Specify which operations should be tracked (e.g. `{'+', '*'}`) and `tqdm` bars will visualise the progress.

## todo

- [x] `sum` & `prod` for non-numeric types (`math.prod` might reject non-numerics; both have `start` argument with unnecessary additional operation; but have better float precision -> automatic fallback?)
- [x] complexities
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
