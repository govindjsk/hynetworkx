from src.hyperedge import Hyperedge, generate_hyperedge, generate_hyperedge_from_vertices

HYPEREDGE_SET_DEFAULTS = {'count': 5}


class HyperedgeSet(set):
    def __init__(self, hyperedges=None):
        hyperedges = hyperedges or ()
        if not all([isinstance(f, Hyperedge) for f in hyperedges]):
            hyperedges = {Hyperedge(f) for f in hyperedges}
        elif not isinstance(hyperedges, set):
            hyperedges = set(hyperedges)
        # assert all([isinstance(f, Hyperedge) for f in hyperedges])
        super(HyperedgeSet, self).__init__(hyperedges)

    @property
    def __summary__(self):
        return '{' + ', '.join([f.__summary__ for f in sorted(self, key=lambda g: (len(g), tuple(sorted(g))))]) + '}'


def generate_hyperedge_set(count=None,
                           size_range=None,
                           id_range=None,
                           label_prefix=None,
                           reserved_vertex_ids=None,
                           reserved_hyperedges=None):
    count = count or HYPEREDGE_SET_DEFAULTS['count']
    F = HyperedgeSet()
    while len(F) < count:
        f = generate_hyperedge(size_range,
                               id_range,
                               label_prefix,
                               reserved_vertex_ids,
                               reserved_hyperedges)
        reserved_hyperedges.add(f)
        F.add(f)
    return F


def generate_hyperedge_set_from_vertices(vertices,
                                         count=None,
                                         size_range=None,
                                         reserved_hyperedges=None):
    count = count or HYPEREDGE_SET_DEFAULTS['count']
    reserved_hyperedges = reserved_hyperedges or set()
    F = HyperedgeSet()
    while len(F) < count:
        f = generate_hyperedge_from_vertices(vertices,
                                             size_range,
                                             reserved_hyperedges)
        reserved_hyperedges.add(f)
        F.add(f)
    return F


def main():
    pass


if __name__ == '__main__':
    main()
