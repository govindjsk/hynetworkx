from random import randrange, sample

import numpy as np

from src.utils import is_iterable
from .vertex_set import VertexSet, generate_vertex_set

HYPEREDGE_DEFAULTS = {'size_range': (1, 5)}


class Hyperedge(frozenset):
    def __new__(cls, elements=()):
        return super(Hyperedge, cls).__new__(Hyperedge, VertexSet(elements))

    # def __init__(self, elements=()):
    #     super(Hyperedge, self).__init__(elements)

    def __repr__(self):
        return 'Hyperedge{{{}}}'.format(', '.join(map(str, list(self))))

    def __str__(self):
        return 'Hyperedge{{{}}}'.format(', '.join(map(str, list(self))))

    def __hash__(self):
        return frozenset.__hash__(self)

    @property
    def __summary__(self):
        return '{' + ', '.join(map(str, sorted([v.id for v in self]))) + '}'

    def get_vector(self, vertex_list=None):
        vector = np.array([int(v in self) for v in vertex_list])
        return vector


def generate_hyperedge(size_range=None,
                       id_range=None,
                       label_prefix=None,
                       reserved_vertex_ids=None,
                       reserved_hyperedges=None):
    size_range = size_range or HYPEREDGE_DEFAULTS['size_range']
    size = randrange(*size_range)
    reserved_hyperedges = set(reserved_hyperedges or [])
    V = generate_vertex_set(size,
                            id_range,
                            label_prefix,
                            reserved_vertex_ids,
                            map(VertexSet, reserved_hyperedges))
    f = Hyperedge(V)
    return f


def generate_hyperedge_from_vertices(vertices,
                                     size_range=None,
                                     reserved_hyperedges=None):
    size_range = size_range or (1, len(vertices) + 1)
    size = randrange(*size_range)
    reserved_hyperedges = set(reserved_hyperedges or [])
    f = Hyperedge(sample(vertices, size))
    if f in reserved_hyperedges:
        f = generate_hyperedge_from_vertices(vertices, size_range, reserved_hyperedges)
    return f


def main():
    pass


if __name__ == '__main__':
    main()
