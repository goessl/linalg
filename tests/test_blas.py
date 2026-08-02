from linalg.blas import *
import numpy as np
import pytest



#preparing values
a = np.array([1, 2, 3], dtype=np.int64)
b = np.array([[5, 6,  7],
              [8, 9, 10]], dtype=np.float64)
c = 11

def test_vneg():
    assert np.allclose(vneg(a), np.array([-1, -2, -3], dtype=np.int64))
    assert np.allclose(vneg(b), np.array([[-5, -6, - 7],
                                          [-8, -9, -10]], dtype=np.float64))

def test_vadd():
    assert np.allclose(vadd(a, b), np.array([[6,  8, 10],
                                             [9, 11, 13]], dtype=np.float64))
    assert np.allclose(vadd(a, c), np.array([12, 13, 14], dtype=np.int64))

def test_vsub():
    assert np.allclose(vsub(a, b), np.array([[-4, -4, -4],
                                             [-7, -7, -7]], dtype=np.float64))
    assert np.allclose(vsub(a, c), np.array([-10, -9, -8], dtype=np.int64))

def test_vmul():
    assert np.allclose(vmul(a, b), np.array([[5, 12, 21],
                                             [8, 18, 30]], dtype=np.float64))
    assert np.allclose(vmul(a, c), np.array([11, 22, 33], dtype=np.int64))
