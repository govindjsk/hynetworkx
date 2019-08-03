from pprint import pprint

import pytest

from src.hyperedge import Hyperedge, generate_hyperedge, Edge
from src.hyperedge_set import HyperedgeSet, generate_hyperedge_set, EdgeSet


def test_init_empty_hyperedge_set():
    F = HyperedgeSet()
    print(F.__summary__)
    assert len(F) == 0


def test_init_trivial_hyperedge_set():
    F = HyperedgeSet({Hyperedge()})
    print(F.__summary__)
    assert len(F) == 1 and len(list(F)[0]) == 0


def test_init_():
    count = 40
    F = generate_hyperedge_set(count=count)
    pprint(F.__summary__)
    assert len(F) == count


def test_edge_set_empty():
    E = EdgeSet()
    print(E.__summary__)
    assert len(E) == 0


@pytest.mark.parametrize('edges', [1,
                                   (1, 2),
                                   [tuple()],
                                   [(1, 2, 3), (1, 2)]])
def test_edge_set_invalid(edges):
    print(edges)
    with pytest.raises(AssertionError):
        EdgeSet(edges)


@pytest.mark.parametrize('edges', [[(1, 2)],
                                   [(1,)],
                                   [(1, 2), (1, 2)],
                                   [Edge([1, 2])]])
def test_edge_set_one(edges):
    print(edges)
    E = EdgeSet(edges)
    print(E.__summary__)
    assert len(E) == 1


def test_edge_set_two():
    E = EdgeSet([(1, 2), (2, 3)])
    print(E.__summary__)
    assert len(E) == 2
