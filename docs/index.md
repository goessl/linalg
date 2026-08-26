# linalg

::: linalg
    options:
      members: false

## Modules

- [`blas`](blas.md)
- [`blas2`](blas2.md)
- [`leibniz`](leibniz.md)
- [`gauss`](gauss.md)
- [`rank`](rank.md)
- [`triangular`](triangular.md)
- [`ortho`](ortho.md)
- [`random`](random.md)
- [`util`](util.md)
- [`progress`](progress.md)

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

- [x] [`blas`](blas.md)
    - [x] `numpy.ufunc`s with callback (same broadcasting, same promotion, ...)
    - [ ] in-place operations
- [x] [`blas2`](blas2.md)
    - [x] `matmul`
    - [x] `outer`
    - [x] `matmulchain`
- [ ] `blas3`
    - [ ] `strassen`
    - [ ] `fft`
- [x] [`progress`](progress.md)
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
    - [x] `pinv`
    - [x] `lstsq`
- [x] `ortho`
    - [x] `gram_schmidt`
    - [ ] `qr`
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
