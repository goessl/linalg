"""Linear algebra with scalar object support & progress visualisation.

```python
>>> import numpy as np
>>> from linalg import *
>>> a, b = np.random.rand(10000), np.random.rand(10000)
>>> c = vadd(a, b)
>>> with visualise('add'):
...     c = vadd(a, b)
...
vadd
add: 100%|████████████████████████████| 10000/10000 [00:00<00:00, 216824.89it/s]
```
"""

from .blas import *
from .progress import *
