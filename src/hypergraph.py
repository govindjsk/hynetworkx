from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix

from src.adjacency_matrix import AdjacencyMatrix
from src.hyperedge_set import HyperedgeSet, generate_hyperedge_set_from_vertices
from src.incidence_matrix import IncidenceMatrix
from src.vertex_set import VertexSet, generate_monotonic_vertex_set
from tqdm import tqdm

HYPERGRAPH_DEFAULTS = {'num_vertices': 5,
                       'num_hyperedges': 10}


class Hypergraph(object):
    def __init__(self, vertices=None, hyperedges=None):
        self.vertices = None
        self.vertex_list = None
        self.hyperedges = None
        self.hyperedge_list = None
        self.incidence_matrix = None
        self.adjacency_matrix = None
        self._S_computed = False
        self._A_computed = False
        print('Initializing V')
        self.V = VertexSet(vertices)
        print('Initializing F')
        self.F = HyperedgeSet(hyperedges)
        print('Computing S')
        self.__compute_incidence_matrix__()
        print('Computing A')
        self.__compute_adjacency_matrix__()

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
        if not self._S_computed:
            self.__compute_incidence_matrix__()
        return self.incidence_matrix

    @property
    def A(self):
        if not self._A_computed:
            self.__compute_adjacency_matrix__()
        return self.adjacency_matrix

    @V.setter
    def V(self, vertices):
        self.vertices = vertices
        self.vertex_list = sorted(self.V)

    @F.setter
    def F(self, hyperedges):
        self.hyperedges = hyperedges
        self.hyperedge_list = sorted(self.F,
                                     key=lambda f: (len(f), tuple(sorted(f))))

    @S.setter
    def S(self, incidence_matrix):
        self.incidence_matrix = incidence_matrix

    @A.setter
    def A(self, adjacency_matrix):
        self.adjacency_matrix = adjacency_matrix

    def __compute_incidence_matrix__(self):
        # indices = []
        list_of_pairlists = [[(v.id, i) for v in self.hyperedge_list[i]]
                             for i in tqdm(range(len(self.hyperedge_list)))]
        pairs = []
        for pairlist in tqdm(list_of_pairlists):
            pairs += pairlist
        # print(len(pairs))
        rows, columns = ([], []) if len(pairs) == 0 else zip(*pairs)
        values = [1]*len(rows)
        matrix = csr_matrix((values, (rows, columns)),
                            shape=(len(self.V), len(self.F)))
        # hyperedge_vectors = [f.get_vector(self.vertex_list)
        #                      for f in tqdm(self.hyperedge_list)]
        # incidence_matrix = IncidenceMatrix(hyperedge_vectors, self.vertex_list)
        incidence_matrix = IncidenceMatrix(matrix, self.vertex_list)
        self._S_computed = True
        self.incidence_matrix = incidence_matrix

    def __compute_adjacency_matrix__(self):
        # adjacency_matrix = AdjacencyMatrix(np.matmul(self.S, self.S.T), self.vertex_list)
        Nr = defaultdict(set)
        for f in tqdm(self.F):
            for v in f:
                Nr[v.id].update({u.id for u in f})
                Nr[v.id].remove(v.id)
        list_of_pairlists = [[(v, u) for u in Nr[v]] for v in Nr]
        pairs = []
        for pairlist in tqdm(list_of_pairlists):
            pairs += pairlist

        rows, columns = ([], []) if len(pairs) == 0 else zip(*pairs)
        print(pairs)
        values = [1] * len(rows)
        matrix = csr_matrix((values, (rows, columns)),
                            shape=(len(self.V), len(self.V)))
        adjacency_matrix = AdjacencyMatrix(matrix, self.vertex_list)
        self._A_computed = True
        self.adjacency_matrix = adjacency_matrix


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
