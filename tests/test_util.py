from linalg.util import *
from fractions import Fraction
import numpy as np
import pytest



#############################################################################
#appended by Claude - the swaps came from test_gauss.py and the predicates
#from test_triangular.py when both moved into linalg.util; the dict helpers
#are new.
#############################################################################

#dictionary accumulation
#`Counter` addition drops non-positive counts, which would silently delete a
#zero category from an announcement; these two keep every key.
def test_dict_add_unions_the_keys():
    assert dict_add({'a': 1, 'b': 2}, {'b': 3, 'c': 4}) \
            == {'a': 1, 'b': 5, 'c': 4}

def test_dict_add_keeps_zero_valued_keys():
    #the whole reason these exist instead of `collections.Counter`
    assert dict_add({'add': 0, 'mul': 5}, {'add': 0, 'mul': 3}) \
            == {'add': 0, 'mul': 8}

def test_dict_add_of_two_empties_is_empty():
    assert dict_add({}, {}) == {}

def test_dict_add_does_not_touch_its_operands():
    a, b = {'x': 1}, {'x': 2}
    dict_add(a, b)
    assert a == {'x': 1} and b == {'x': 2}

def test_dict_add_returns_a_new_mapping():
    a = {'x': 1}
    assert dict_add(a, {}) is not a

def test_dict_iadd_accumulates_in_place_and_returns_the_target():
    a = {'a': 1, 'b': 2}
    r = dict_iadd(a, {'b': 3, 'c': 4})
    assert r is a and a == {'a': 1, 'b': 5, 'c': 4}

def test_dict_iadd_keeps_zero_valued_keys():
    a = {'add': 0, 'mul': 5}
    dict_iadd(a, {'add': 0, 'mul': 3})
    assert a == {'add': 0, 'mul': 8}

def test_dict_iadd_does_not_touch_its_source():
    a, b = {'x': 1}, {'x': 2, 'y': 3}
    dict_iadd(a, b)
    assert b == {'x': 2, 'y': 3}

def test_dict_add_accepts_a_read_only_mapping():
    #`matmulchain_plan` hands out `MappingProxyType`s
    from types import MappingProxyType
    assert dict_add(MappingProxyType({'a': 1}), {'a': 2}) == {'a': 3}


def test_dict_sub_unions_the_keys():
    assert dict_sub({'a': 5, 'b': 2}, {'b': 3, 'c': 4}) \
            == {'a': 5, 'b': -1, 'c': -4}

def test_dict_sub_keeps_zero_valued_keys():
    #a category that cancels exactly must stay, or the announcement loses it
    assert dict_sub({'add': 4, 'mul': 5}, {'add': 4, 'mul': 3}) \
            == {'add': 0, 'mul': 2}

def test_dict_sub_of_two_empties_is_empty():
    assert dict_sub({}, {}) == {}

def test_dict_sub_does_not_touch_its_operands():
    a, b = {'x': 1}, {'x': 2}
    dict_sub(a, b)
    assert a == {'x': 1} and b == {'x': 2}

def test_dict_sub_returns_a_new_mapping():
    a = {'x': 1}
    assert dict_sub(a, {}) is not a

def test_dict_sub_treats_a_missing_key_as_zero():
    #`pinv` subtracts two costs whose category sets need not coincide
    assert dict_sub({}, {'mul': 3}) == {'mul': -3}

def test_dict_isub_subtracts_in_place_and_returns_the_target():
    a = {'a': 5, 'b': 2}
    r = dict_isub(a, {'b': 3, 'c': 4})
    assert r is a and a == {'a': 5, 'b': -1, 'c': -4}

def test_dict_isub_keeps_zero_valued_keys():
    a = {'add': 4, 'mul': 5}
    dict_isub(a, {'add': 4, 'mul': 3})
    assert a == {'add': 0, 'mul': 2}

def test_dict_isub_does_not_touch_its_source():
    a, b = {'x': 1}, {'x': 2, 'y': 3}
    dict_isub(a, b)
    assert b == {'x': 2, 'y': 3}

def test_dict_sub_accepts_a_read_only_mapping():
    from types import MappingProxyType
    assert dict_sub(MappingProxyType({'a': 3}), {'a': 2}) == {'a': 1}

def test_dict_sub_undoes_dict_add():
    #`pinv` leans on this: it adds costs up, then subtracts one back out
    a, b = {'add': 7, 'mul': 0}, {'mul': 4, 'truediv': 2}
    assert dict_sub(dict_add(a, b), b) == {'add': 7, 'mul': 0, 'truediv': 0}



#swaps
def test_swap_rows():
    A = np.arange(6).reshape(3, 2)
    swap_rows(A, 0, 2)
    assert np.array_equal(A, [[4, 5], [2, 3], [0, 1]])

def test_swap_columns():
    A = np.arange(6).reshape(2, 3)
    swap_columns(A, 0, 2)
    assert np.array_equal(A, [[2, 1, 0], [5, 4, 3]])

def test_swap_pivot_swaps_both():
    A = np.arange(9).reshape(3, 3)
    swap_pivot(A, 0, 1, 2)
    #row 0 <-> row 1, then column 0 <-> column 2
    assert np.array_equal(A, [[5, 4, 3], [2, 1, 0], [8, 7, 6]])

@pytest.mark.parametrize('f', [swap_rows, swap_columns])
def test_swap_with_itself_changes_nothing(f):
    A = np.arange(6).reshape(3, 2)
    B = A.copy()
    f(A, 1, 1)
    assert np.array_equal(A, B)

@pytest.mark.parametrize('f', [swap_rows, swap_columns])
def test_swap_rejects_a_non_array(f):
    with pytest.raises(TypeError):
        f([[1, 2], [3, 4]], 0, 1)

def test_swap_pivot_rejects_a_non_array():
    with pytest.raises(TypeError):
        swap_pivot([[1, 2], [3, 4]], 0, 1, 1)

@pytest.mark.parametrize('f', [swap_rows, swap_columns])
def test_swap_rejects_bad_shapes(f):
    with pytest.raises(ValueError):
        f(np.zeros(3), 0, 1)

def test_swap_pivot_rejects_bad_shapes():
    with pytest.raises(ValueError):
        swap_pivot(np.zeros(3), 0, 1, 1)

def test_swaps_are_exact_for_fractions():
    #the swaps are the one in-place step of every pivoting algorithm here
    A = np.array([[Fraction(1, 3), Fraction(1, 6)],
                  [Fraction(1, 2), Fraction(1, 5)]], object)
    swap_rows(A, 0, 1)
    assert A[0, 0] == Fraction(1, 2) and A[1, 0] == Fraction(1, 3)



#predicates
def test_is_perm_accepts_permutation_matrices():
    assert is_perm(np.eye(3, dtype=int))
    assert is_perm([[0, 1, 0], [0, 0, 1], [1, 0, 0]])

@pytest.mark.parametrize('P', [
    [[2, 0], [0, 2]],   #ones scaled away
    [[1, 1], [0, 0]],   #column sums wrong
    [[1, 0], [1, 0]],   #row sums wrong
])
def test_is_perm_rejects_non_permutations(P):
    assert not is_perm(P)

def test_is_tril_and_is_triu():
    assert is_tril([[1, 0], [2, 3]]) and not is_triu([[1, 0], [2, 3]])
    assert is_triu([[1, 2], [0, 3]]) and not is_tril([[1, 2], [0, 3]])

def test_a_diagonal_matrix_is_both_triangular():
    #the predicates look strictly above/below, so the diagonal is shared
    assert is_tril(np.eye(4)) and is_triu(np.eye(4))

def test_the_predicates_take_the_non_square_overhang():
    #the factors are routinely oblong, so this is the shape they see
    assert is_tril([[1, 0, 0], [2, 3, 0]]) and is_triu([[1, 2, 3], [0, 4, 5]])

@pytest.mark.parametrize('predicate', [is_perm, is_tril, is_triu])
def test_the_predicates_accept_array_likes(predicate):
    predicate([[1, 0], [0, 1]])

@pytest.mark.parametrize('A', [np.zeros(3), np.zeros((2, 2, 2))])
@pytest.mark.parametrize('predicate', [is_perm, is_tril, is_triu])
def test_the_predicates_reject_bad_dimensions(predicate, A):
    with pytest.raises(ValueError):
        predicate(A)

def test_is_perm_rejects_a_non_square_matrix():
    #unlike the triangular pair, a permutation matrix has to be square
    with pytest.raises(ValueError):
        is_perm(np.zeros((2, 3)))

@pytest.mark.parametrize('predicate', [is_perm, is_tril, is_triu])
def test_an_empty_matrix_satisfies_every_predicate(predicate):
    #vacuously - and an empty matrix does come back out of all four
    assert predicate(np.empty((0, 0)))
