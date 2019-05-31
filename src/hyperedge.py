from src.vertex_frozenset import VertexFrozenset
from src.vertex_set import VertexSet, generate_vertex_set


class Hyperedge(frozenset):

    def __new__(cls, elements):
        return super(Hyperedge, cls).__new__(Hyperedge, VertexSet(elements))

    # def __init__(self, elements=()):
    #     super(Hyperedge, self).__init__(elements)

    def __repr__(self):
        return 'Hyperedge{{{}}}'.format(', '.join(map(str, list(self))))

    def __str__(self):
        return 'Hyperedge{{{}}}'.format(', '.join(map(str, list(self))))

    # def __hash__(self):
    #     return hash(frozenset(self))


def generate_hyperedge(size=None,
                       id_range=None,
                       label_prefix=None,
                       reserved_ids=None):
    V = generate_vertex_set(size, id_range, label_prefix, reserved_ids)
    f = Hyperedge(V)
    return f


def main():
    pass


if __name__ == '__main__':
    main()
