# linalg

Linear algebra with scalar object support & progress visualisation.

```python
>>> import numpy as np
>>> from linalg import *
>>> a, b = np.random.rand(10000), np.random.rand(10000)
>>> c = vadd(a, b)
>>> c = vadd(a, b, progress=True)
add: 100%|████████████████████████████| 10000/10000 [00:00<00:00, 216824.89it/s]
```

## Installation

```console
pip install git+https://github.com/goessl/linalg.git
```

## Usage

**Enjoy the [documentation webpage](https://goessl.github.io/linalg).**


## Dependencies

- [`numpy`](https://numpy.org/)
- [`tqdm`](https://tqdm.github.io/)

## Roadmap

- [x] Deploy
- [x] Production
- [ ] [`blas`](linalg/blas.py) & [`blas2`](linalg/blas2.py)
    - [x] `numpy.ufunc`s with callback (same broadcasting, same promotion, ...)
    - [ ] in-place operations
    - [x] `dot`
    - [x] `outer`
    - [x] `matmul`
    - [ ] `mat vec mul`
- [x] [`progress`](linalg/progress.py)
    - [x] progress visualisation in ~~context block~~ decorator
    - [ ] complexity tracking for functions in development
- [x] `leibniz`
    - [x] `det_leibniz`
- [ ] `laplace`
    - [ ] `det_laplace`
- [ ] `gauss`
    - [ ] `det_gauss`
    - [ ] `inv_gauss`
    - [ ] `bareiss`
- [ ] `adjugate`
    - [ ] `minor`
    - [ ] `cofactor`
    - [ ] `adj`
    - [ ] `cof`
- [ ] `lu`
    - [ ] `lu`
    - [ ] `plu`
    - [ ] `luq`
    - [ ] `pluq`
    - [ ] `banachiewicz`
- [ ] `rank`
    - [ ] `ref_gauss`
    - [ ] `rank_decomp`
    - [ ] `pinv`
    - [ ] `lstsq`
- [ ] `faddeev_leverrier`
    - [ ] `det_faddeev_leverrier`
    - [ ] `charp_faddeev_leverrier`
    - [ ] `adj_faddeev_leverrier`
- [x] Ballin

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
