import os
from collections import defaultdict
from itertools import combinations

import networkx as nx
from scipy.sparse import csr_matrix
from src.adjacency_matrix import AdjacencyMatrix
from src.hyperedge import Edge
from src.hyperedge_set import HyperedgeSet, generate_hyperedge_set_from_vertices, EdgeSet, \
    generate_edge_set_from_vertices
from src.incidence_matrix import IncidenceMatrix
from src.vertex_set import VertexSet, generate_monotonic_vertex_set
from tqdm import tqdm

HYPERGRAPH_DEFAULTS = {'num_vertices': 5,
                       'num_hyperedges': 10,
                       'num_edges': 10}
DATA_DEFAULTS = {'base_path_benson': '/home/govindjsk/repos/data/benson/'}


class Hypergraph(object):
    def __init__(self, vertices=None, hyperedges=None):
        self.vertices = None
        self.vertex_list = None
        self.hyperedge_list = None
        self.incidence_matrix = None
        self.adjacency_matrix = None
        self.induced_graph = None
        self.hyperedges = None
        self._S_computed = False
        self._A_computed = False
        self._G_computed = False
        print('Initializing V')
        self.V = VertexSet(vertices)
        print('Initializing F')
        self.F = HyperedgeSet(hyperedges)
        # print('Computing S')
        # self.__compute_incidence_matrix__()
        # print('Computing A')
        # self.__compute_adjacency_matrix__()

    def __repr__(self):
        return 'Hypergraph(\n' \
               '\t{},\n' \
               '\t{})'.format(self.V, self.F)

    @property
    def __summary__(self):
        return '({}, {})'.format('{' + ', '.join(map(str, self.V.__summary__)) + '}',
                                 self.F.__summary__)

    @property
    def H(self):
        return tuple((self.V, self.F))

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

    @property
    def G(self):
        if not self._G_computed:
            self.__compute_induced_graph__()
        return self.induced_graph

    @property
    def E(self):
        return self.G.E

    @V.setter
    def V(self, vertices):
        self.vertices = vertices
        self.vertex_list = sorted(self.V)

    @F.setter
    def F(self, hyperedges):
        self.hyperedges = hyperedges
        self.hyperedge_list = sorted(self.F,
                                     key=lambda f: (len(f), tuple(sorted(f))))

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
                            shape=(max(self.V)[0]+1, max(columns)+1))
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
        # print(pairs)
        values = [1] * len(rows)
        matrix = csr_matrix((values, (rows, columns)),
                            shape=(max(self.V)[0]+1, max(self.V)[0]+1))
        adjacency_matrix = AdjacencyMatrix(matrix, self.vertex_list)
        self._A_computed = True
        self.adjacency_matrix = adjacency_matrix

    def __compute_induced_graph__(self):
        E = set()
        for f in self.F:
            if len(f) < 2:
                continue
            if len(f) == 2:
                E.add(frozenset(f))
                continue
            es = set(map(Edge, combinations(f, 2)))
            E.update(es)
        self.induced_graph = Graph(self.V, E)
        self._G_computed = True


class Graph(Hypergraph):
    def __init__(self, vertices=None, edges=None):
        super(Graph, self).__init__(vertices, edges)
        self.edges = EdgeSet(set(map(Edge, self.F)))

    def __repr__(self):
        return 'Graph(\n' \
               '\t{},\n' \
               '\t{})'.format(self.V, self.E)

    @property
    def E(self):
        return self.edges

    def to_nx(self):
        nx_G = nx.Graph()
        nx_V = {v.id for v in self.V}
        nx_E = [[v.id for v in e] for e in self.E]
        nx_G.add_nodes_from(nx_V)
        nx_G.add_edges_from(nx_E)
        return nx_G


def generate_hypergraph(num_vertices=None, num_hyperedges=None, size_range=None):
    num_vertices = num_vertices or HYPERGRAPH_DEFAULTS['num_vertices']
    num_hyperedges = num_hyperedges or HYPERGRAPH_DEFAULTS['num_hyperedges']
    V = generate_monotonic_vertex_set(num_vertices)
    F = generate_hyperedge_set_from_vertices(V, num_hyperedges, size_range=size_range)
    H = Hypergraph(V, F)
    return H


def generate_graph(num_vertices=None, num_edges=None):
    num_vertices = num_vertices or HYPERGRAPH_DEFAULTS['num_vertices']
    num_edges = num_edges or HYPERGRAPH_DEFAULTS['num_edges']
    V = generate_monotonic_vertex_set(num_vertices)
    E = generate_edge_set_from_vertices(V, num_edges)
    G = Graph(V, E)
    return G


def parse_benson_hypergraph(name, base_path=None, ignore_time=True):
    base_path = base_path or DATA_DEFAULTS['base_path_benson']
    path = os.path.join(base_path, name)
    nverts_path = os.path.join(path, name + '-nverts.txt')
    simplices_path = os.path.join(path, name + '-simplices.txt')
    nverts = [int(l.rstrip('\n')) for l in open(nverts_path, 'r')]
    # TODO: Remember that we are reindexing vertices to 0-index.
    #  This has to be followed while initializing labels as well.
    simplices = [int(l.rstrip('\n'))-1 for l in open(simplices_path, 'r')]
    times = None
    if not ignore_time:
        times_path = os.path.join(path, name + '-times.txt')
        times = [l.rstrip('\n') for l in open(times_path, 'r')]
    labels_path = os.path.join(path, name + '-node-labels.txt')
    try:
        labels = [l.rstrip('\n') for l in open(labels_path, 'r')]
    except FileNotFoundError:
        print('No labels found.')
        labels = None
    vertices = list(sorted(set(simplices)))
    hyperedges = []
    curr = 0
    for nv in nverts:
        simplices = simplices[curr:]
        hyperedges.append(simplices[:nv])
        curr += nv
    H = Hypergraph(vertices, hyperedges)
    return H


def main():
    pass


if __name__ == '__main__':
    main()
