from src.hyperedge_set import HyperedgeSet, generate_hyperedge_set_from_vertices
from src.incidence_matrix import IncidenceMatrix
from src.vertex_set import VertexSet, generate_monotonic_vertex_set

HYPERGRAPH_DEFAULTS = {'num_vertices': 5,
                       'num_hyperedges': 10}


class Hypergraph(object):
    def __init__(self, vertices=None, hyperedges=None):
        self.vertices = None
        self.hyperedges = None
        self.incidence_matrix = None
        self.V = VertexSet(vertices)
        self.F = HyperedgeSet(hyperedges)
        self.__compute_incidence_matrix__()

    def __repr__(self):
        return 'Hypergraph({}, {})'.format(self.V, self.F)

    @property
    def __summary__(self):
        return '({}, {})'.format('{' + ', '.join(map(str, self.V.__summary__)) + '}',
                                 self.F.__summary__)

    @property
    def V(self):
        return self.vertices

    @property
    def F(self):
        return self.hyperedges

    @property
    def S(self):
        self.__compute_incidence_matrix__()
        return self.incidence_matrix

    @V.setter
    def V(self, vertices):
        self.vertices = vertices

    @F.setter
    def F(self, hyperedges):
        self.hyperedges = hyperedges

    @S.setter
    def S(self, incidence_matrix):
        self.incidence_matrix = incidence_matrix

    def __compute_incidence_matrix__(self):
        vertex_list = sorted(self.V)
        hyperedge_vectors = [f.get_vector(vertex_list)
                             for f in sorted(self.F,
                                             key=lambda g: (len(g), tuple(sorted(g))))]
        incidence_matrix = IncidenceMatrix(hyperedge_vectors, vertex_list)
        self.S = incidence_matrix.T


def generate_hypergraph(num_vertices=None, num_hyperedges=None):
    num_vertices = num_vertices or HYPERGRAPH_DEFAULTS['num_vertices']
    num_hyperedges = num_hyperedges or HYPERGRAPH_DEFAULTS['num_hyperedges']
    V = generate_monotonic_vertex_set(num_vertices)
    F = generate_hyperedge_set_from_vertices(V, num_hyperedges)
    H = Hypergraph(V, F)
    return H


def main():
    pass


if __name__ == '__main__':
    main()
