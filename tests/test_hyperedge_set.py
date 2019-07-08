from pprint import pprint

from src.hyperedge import Hyperedge, generate_hyperedge
from src.hyperedge_set import HyperedgeSet, generate_hyperedge_set


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
