import os

from tqdm import tqdm

from src.adjacency_matrix import AdjacencyMatrix
from .utils import print_matrix, _or, get_printable_matrix
from .vertex import Vertex
from scipy.sparse import csr_matrix

DATA_DEFAULTS = {'base_path_benson': '/home/govindjsk/repos/data/benson/hypergraphs/'}


class IncidenceMatrix(csr_matrix):
    # def __new__(cls, matrix=None, vertex_list=None):
    #     matrix = _or(matrix, csr_matrix, ([],))
    #     obj = csr_matrix(matrix).view(cls)
    #     obj.vertex_list = vertex_list
    #     if len(obj.shape) == 1:
    #         if obj.shape[0] == 0:
    #             obj = obj.reshape(0, 0)
    #         else:
    #             obj = obj.reshape(obj.shape[0], 1)
    #     return obj

    # def __array_finalize__(self, obj):
    #     if obj is None:
    #         return
    #     self.vertex_list = getattr(obj, 'vertex_list', None)

    def __init__(self, matrix=None, vertex_list=None, **kwargs):
        matrix = _or(matrix, csr_matrix, ([],))
        self.vertex_list = vertex_list
        self.adjacency_matrix = None
        super(IncidenceMatrix, self).__init__(matrix, **kwargs)

    @property
    def A(self):
        if self.adjacency_matrix is None:
            self.adjacency_matrix = self.__compute_adjacency_matrix__()
        return AdjacencyMatrix(self.adjacency_matrix, self.vertex_list)

    def pprint(self, show_labels=False):
        if show_labels:
            vertex_list = self.vertex_list or [Vertex(i) for i in range(self.shape[0])]
            row_labels = [v.label for v in vertex_list]
            print_matrix(self, row_labels)
        else:
            print_matrix(self)

    def __compute_adjacency_matrix__(self):
        A = self*self.T
        A.setdiag(0)
        A[A > 0] = 1
        return A
    # def __repr__(self):
    #     matrix = self[:5, :5]
    #     return str(matrix.to_dense())

    def __repr__(self):
        return 'A {} x {} incidence matrix with nnz = {}\n'.format(*self.shape, self.nnz) + \
               get_printable_matrix(self[:5, :5], self.vertex_list[:5])


def get_benson_incidence_matrix(name, base_path=None, ignore_time=True, force_parse=False):
    base_path = base_path or DATA_DEFAULTS['base_path_benson']
    path = os.path.join(base_path, name)
    S_path = os.path.join(path, name + '-S.pkl')
    from pickle import load, dump
    try:
        assert not force_parse, 'Parsing forced'
        print('Attempting to read cache...')
        f = open(S_path, 'rb')
        S = load(f)
        print('Cache found!')
    except (FileNotFoundError, EOFError, AssertionError):
        print('Cache not found or parsing forced. Reading from scratch...')
        f = open(S_path, 'wb')
        S = parse_benson_incidence_matrix(name, base_path, ignore_time)
        dump(S, f)
    f.close()
    return S


def parse_benson_incidence_matrix(name, base_path=None, ignore_time=True):
    base_path = base_path or DATA_DEFAULTS['base_path_benson']
    path = os.path.join(base_path, name)
    nverts_path = os.path.join(path, name + '-nverts.txt')
    simplices_path = os.path.join(path, name + '-simplices.txt')
    print('Reading nverts...')
    nverts = [int(l.rstrip('\n')) for l in tqdm(open(nverts_path, 'r'))]
    # TODO: Remember that we are reindexing vertices to 0-index.
    #  This has to be followed while initializing labels as well.
    print('Reading simplices...')
    simplices = [int(l.rstrip('\n')) - 1 for l in tqdm(open(simplices_path, 'r'))]
    labels_path = os.path.join(path, name + '-node-labels.txt')
    vertices = list(sorted(set(simplices)))
    try:
        print('Reading labels...')
        labels = [l.rstrip('\n') for l in tqdm(open(labels_path, 'r'))]
        labels = [' '.join(l.split(' ')[1:]) for l in labels]
    except FileNotFoundError:
        print('No labels found.')
        labels = ['vertex_{}'.format(v) for v in vertices]
    n = max(vertices) + 1
    hyperedges = set()
    rows = []
    cols = []
    j = 0
    print('Parsing simplices...')
    i = 0
    for nv in tqdm(nverts):
        hyperedge = frozenset(simplices[i: i + nv])
        if hyperedge not in hyperedges:
            hyperedges.add(hyperedge)
            rows.extend(list(hyperedge))
            cols.extend([j] * len(hyperedge))
            j += 1
        i += nv
    m = len(hyperedges)
    print('Creating hypergraph...')
    print(len(rows), len(cols), n, m)
    matrix = csr_matrix(([1] * len(rows), (rows, cols)), shape=(n, m))
    vertex_list = [Vertex(i, labels[i]) for i in range(len(labels))]
    S = IncidenceMatrix(matrix, vertex_list)
    return S


def main():
    S = parse_benson_incidence_matrix('coauth-DBLP')
    print(S.shape)


if __name__ == '__main__':
    main()
