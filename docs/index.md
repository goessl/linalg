# linalg

::: linalg
    options:
      members: false

## Modules

### [`blas`](blas.md)

- [`ufunc_with_cb`][linalg.blas.ufunc_with_cb]
- [`vpos`][linalg.blas.vpos]
- [`vneg`][linalg.blas.vneg]
- [`vadd`][linalg.blas.vadd]
- [`vsub`][linalg.blas.vsub]
- [`vmul`][linalg.blas.vmul]
- [`vtruediv`][linalg.blas.vtruediv]
- [`vfloordiv`][linalg.blas.vfloordiv]
- [`vmod`][linalg.blas.vmod]

### [`blas2`](blas2.md)

- [`matmul`][linalg.blas2.matmul]
- [`outer`][linalg.blas2.outer]
- [`matmulchain`][linalg.blas2.matmulchain]

### [`leibniz`](leibniz.md)

- [`permutations`][linalg.leibniz.permutations]
- [`det_leibniz`][linalg.leibniz.det_leibniz]

### [`gauss`](gauss.md)

- [`det_gauss`][linalg.gauss.det_gauss]
- [`inv_gauss`][linalg.gauss.inv_gauss]
- [`is_ref`][linalg.gauss.is_ref]
- [`ref_gauss`][linalg.gauss.ref_gauss]

### [`rank`](rank.md)

- [`rank_decomp`][linalg.rank.rank_decomp]
- [`nullspace`][linalg.rank.nullspace]
- [`pinv`][linalg.rank.pinv]
- [`lstsq`][linalg.rank.lstsq]

### [`triangular`](triangular.md)

- [`is_perm`][linalg.triangular.is_perm]
- [`is_tril`][linalg.triangular.is_tril]
- [`is_triu`][linalg.triangular.is_triu]
- [`lu`][linalg.triangular.lu]
- [`plu`][linalg.triangular.plu]
- [`luq`][linalg.triangular.luq]
- [`pluq`][linalg.triangular.pluq]

### [`ortho`](ortho.md)

- [`are_orthogonal`][linalg.ortho.are_orthogonal]
- [`are_normalised`][linalg.ortho.are_normalised]
- [`are_orthonormal`][linalg.ortho.are_orthonormal]
- [`is_orthogonal`][linalg.ortho.is_orthogonal]
- [`is_semiorthogonal`][linalg.ortho.is_semiorthogonal]
- [`is_orthonormal`][linalg.ortho.is_orthonormal]
- [`is_semiorthonormal`][linalg.ortho.is_semiorthonormal]
- [`gram_schmidt`][linalg.ortho.gram_schmidt]
- [`qr_decomp`][linalg.ortho.qr_decomp]

### [`random`](random.md)

- [`randz`][linalg.random.randz]
- [`randq`][linalg.random.randq]
- [`vrandz`][linalg.random.vrandz]
- [`vrandq`][linalg.random.vrandq]
- [`mrandqr`][linalg.random.mrandqr]

### [`util`](util.md)

- [`dict_add`][linalg.util.dict_add]
- [`dict_iadd`][linalg.util.dict_iadd]
- [`dict_sub`][linalg.util.dict_sub]
- [`dict_isub`][linalg.util.dict_isub]
- [`swap_rows`][linalg.util.swap_rows]
- [`swap_columns`][linalg.util.swap_columns]
- [`swap_pivot`][linalg.util.swap_pivot]

### [`progress`](progress.md)

- [`Progress`][linalg.progress.Progress]
- [`visualisable`][linalg.progress.visualisable]

## Installation

```console
pip install git+https://github.com/goessl/linalg.git
```

### Building the documentation

```console
mkdocs build --clean --strict
```

## Dependencies

- [`numpy`](https://numpy.org/)
- [`tqdm`](https://tqdm.github.io/)

## Roadmap

### Modules

- [x] `blas`
    - [x] `numpy.ufunc`s with callback (same broadcasting, same promotion, ...)
    - [ ] in-place operations
- [x] `blas2`
    - [x] `matmul`
    - [x] `outer`
    - [x] `matmulchain`
- [ ] `blas3`
    - [ ] `strassen`
    - [ ] `fft`
- [x] `progress`
    - [x] progress visualisation in ~~context block~~ decorator
    - [ ] complexity tracking for functions in development
- [x] `leibniz`
    - [x] `det_leibniz`
- [ ] `laplace`
    - [ ] `det_laplace`
- [x] `gauss`
    - [x] `det_gauss`
    - [x] `inv_gauss`
    - [x] `ref_gauss` & `is_ref`
    - [ ] `bareiss`
- [ ] `adjugate`
    - [ ] `minor`
    - [ ] `cofactor`
    - [ ] `adj`
    - [ ] `cof`
- [x] `triangular`
    - [x] `lu`
    - [x] `plu`
    - [x] `luq`
    - [x] `pluq`
    - [ ] `banachiewicz`
- [x] `rank`
    - [x] `rank_decomp`
    - [x] `nullspace`
    - [x] `pinv`
    - [x] `lstsq`
- [x] `ortho`
    - [x] `gram_schmidt`
    - [x] `qr`
    - [ ] complex support
- [x] `random`
    - [x] `randz`, `randq`, `vrandz`, `vrandq`
    - [x] `mrandqr`
- [ ] `faddeev_leverrier`
    - [ ] `det_faddeev_leverrier`
    - [ ] `charp_faddeev_leverrier`
    - [ ] `adj_faddeev_leverrier`
- [x] `util`
- [ ] `order`
    Sorting algorithms

### Package

- [x] deploy
- [x] production
- [x] ballin
- [ ] coding conventions
- [x] zero and one parameters for type safety
- [ ] no unnecessary `a_ij/a_ij` divisions like in `det_gauss`
- [ ] complete edge case testing (mainly array side lengths 0)
- [ ] decorator should also prepend sanitiser to announcer to be safely usable
- [ ] better annotations
    - [ ] `visualisable` wrapping
    - [ ] `mkdocstrings` annotations still wrong default rendering despite
    overwrite from docstring
- [ ] links from this list to the actual functions

## License (MIT)

Copyright (c) 2026 Sebastian Gössl

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
