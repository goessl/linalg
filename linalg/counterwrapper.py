import numpy as np
from collections import Counter



class CounterWrapper:
    counter = Counter()
    
    def __init__(self, v):
        self.v = v
    
    
    def __float__(self):
        return float(self.v)
    
    def __bool__(self):
        return bool(self.v)
    
    def __eq__(self, other):
        if isinstance(other, CounterWrapper):
            return self.v == other.v
        return self.v == other
    
    def __repr__(self):
        return f'CounterWrapper({self.v})'
    
    
    def __abs__(self):
        return abs(self.v)
    
    def __pos__(self):
        return self
    
    def __neg__(self):
        return CounterWrapper(-self.v)
    
    
    def __add__(self, other):
        if not (isinstance(other, int) and other==0):
            CounterWrapper.counter['+'] += 1 #don't count +int(0)
        return CounterWrapper(self.v + (other.v if isinstance(other, CounterWrapper) else other))
    
    def __radd__(self, other):
        if not (isinstance(other, int) and other==0):
            CounterWrapper.counter['+'] += 1 #don't count +int(0)
        return CounterWrapper(other + self.v)
    
    def __sub__(self, other):
        if not (isinstance(other, int) and other==0):
            CounterWrapper.counter['+'] += 1 #don't count -int(0)
        return CounterWrapper(self.v - (other.v if isinstance(other, CounterWrapper) else other))
    
    def __rsub__(self, other):
        if not (isinstance(other, int) and other==0):
            CounterWrapper.counter['+'] += 1 #don't count int(0)-
        return CounterWrapper(other - self.v)
    
    def __mul__(self, other):
        if not (isinstance(other, int) and other in {-1, 0, +1}):
            CounterWrapper.counter['*'] += 1 #don't count *int(-1), *int(0), *int(+1)
        return CounterWrapper(self.v * (other.v if isinstance(other, CounterWrapper) else other))
    
    def __rmul__(self, other):
        if not (isinstance(other, int) and other in {-1, 0, +1}):
            CounterWrapper.counter['*'] += 1 #don't count *int(-1), *int(0), *int(+1)
        return CounterWrapper(other * self.v)
    
    def __truediv__(self, other):
        if not (isinstance(other, int) and other in {-1, +1}):
            CounterWrapper.counter['/'] += 1 #don't count /int(-1), /int(+1)
        return CounterWrapper(self.v / (other.v if isinstance(other, CounterWrapper) else other))
    
    def __rtruediv__(self, other): #don't count int(0)/
        if not (isinstance(other, int) and other==0):
            CounterWrapper.counter['/'] += 1
        return CounterWrapper(other / self.v)
    
    def __floordiv__(self, other):
        if not (isinstance(other, int) and other in {-1, +1}):
            CounterWrapper.counter['//'] += 1 #don't count //int(-1), //int(+1)
        return CounterWrapper(self.v // (other.v if isinstance(other, CounterWrapper) else other))
    
    def __rfloordiv__(self, other): #don't count int(0)//
        if not (isinstance(other, int) and other==0):
            CounterWrapper.counter['//'] += 1
        return CounterWrapper(other // self.v)


def toCounterWrappers(a):
    return np.vectorize(CounterWrapper, otypes=[object])(a)

def fromCounterWrappers(a):
    return np.vectorize(lambda c: c.v if isinstance(c, CounterWrapper) else c, otypes=[object])(a)
