from linalg.counterwrapper import CounterWrapper



f = CounterWrapper(1)
g = CounterWrapper(2)
f + g
assert CounterWrapper.counter['+'] == 1
f + 0
assert CounterWrapper.counter['+'] == 1
