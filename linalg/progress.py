from tqdm.auto import tqdm
from functools import reduce



class Progress:
    """Progress handler for algorithms.
    
    Use as context. Total number of operations must be known beforehand.
    Call `update(op, n=1)` to increment tracking.
    """
    
    def __init__(self, totals, descprefix=''):
        """Create a new progress handler.
        
        `totals` should be a dictionary with the tracked operations as keys
        and the total number of operations as values.
        """
        self.pbars = \
                {o:tqdm(desc=descprefix+o, total=t) for o, t in totals.items()}
    
    def __enter__(self):
        return self
    
    def __exit__(self, type, value, traceback):
        for pbar in self.pbars.values():
            pbar.close()
    
    def update(self, op, n=1):
        """Increment the operation `op` progress by `n`.
        
        If `op` is not tracked nothing happens.
        """
        if op in self.pbars:
            self.pbars[op].update(n)
    
    def add(self, a, b):
        c = a + b
        self.update('+')
        return c
    
    def mul(self, a, b):
        c = a * b
        self.update('*')
        return c
    
    def sum(self, iterable):
        return reduce(self.add, iterable)
    
    def prod(self, iterable):
        return reduce(self.mul, iterable)
