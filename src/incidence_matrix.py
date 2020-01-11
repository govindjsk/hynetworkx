import os

from collections import defaultdict
from tqdm import tqdm
import numpy as np
from src.adjacency_matrix import AdjacencyMatrix
from src.data_preparer import filter_size, filter_time
from .utils import print_matrix, _or, get_printable_matrix, get_base_path
from .vertex import Vertex
from scipy.sparse import csr_matrix
import networkx as nx
from joblib import Memory
data_path = get_base_path()
cachedir = os.path.join(data_path, 'cache')
memory = Memory(cachedir, verbose=0)

DATA_DEFAULTS = {'base_path_benson': '/home/govinds/Downloads/'}


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


@memory.cache
def parse_benson_incidence_matrix(name,base_path=None,split_mode=None,max_size_limit=None,st_time=None,en_time=None):
    
    ase_path = base_path or DATA_DEFAULTS['base_path_benson']
    path = os.path.join(base_path, name)  
    
    try:
        file_name = os.path.join(path, name + '_native.edges')
        f = open(file_name, "r")
        edgelist=[]
        for line in f:
            a = int(line.split(' ')[0].strip())  # \t
            b = int(line.split(' ')[1].strip())
            if (a!= b):
                edge=[a,b]
                edgelist.append(edge)
        graph = nx.Graph()
        graph.add_edges_from(edgelist)
        A=nx.adj_matrix(graph)
#         A=adj
#         test_pairs = list(zip(*triu(A).nonzero()))
#         print(len(test_pairs))
        G = nx.from_scipy_sparse_matrix(A)
        S=nx.incidence_matrix(G)
        times=[]
        for i in range(len(edgelist)):
            times.append(0)
        id_label_map={}
        f.close()
        return S, np.array(times), id_label_map
    
    except IOError:
        nverts_path = os.path.join(path, name + '-nverts.txt')
        simplices_path = os.path.join(path, name + '-simplices.txt')
        times_path = os.path.join(path, name + '-times.txt')
        print('Reading nverts...')
        nverts = [int(l.rstrip('\n')) for l in tqdm(open(nverts_path, 'r'))]
        # TODO: Remember that we are reindexing vertices to 0-index.
        #  This has to be followed while initializing labels as well.
        print('Reading simplices...')
        simplices = [int(l.rstrip('\n')) - 1 for l in tqdm(open(simplices_path, 'r'))]
        labels_path = os.path.join(path, name + '-node-labels.txt')
        vertices = list(sorted(set(simplices)))
        n = max(vertices) + 1
        try:
            print('Reading labels...')
            labels = [l.rstrip('\n') for l in tqdm(open(labels_path, 'r'))]
            labels = [' '.join(l.split(' ')[1:]) for l in labels]
        except FileNotFoundError:
            print('No labels found.')
            labels = ['vertex_{}'.format(i) for i in range(n)]
        print('Reading times...')
        times = [float(l.rstrip('\n')) for l in tqdm(open(times_path, 'r'))]
        hyperedges = set()
        hyperedge_times_map = defaultdict(set)
        hyperedge_list = []
        rows = []
        cols = []
        j = 0
        print('Parsing simplices...')
        i = 0
        iterator = list(zip(nverts, times))
        for nv, time in tqdm(iterator):
            hyperedge = frozenset(simplices[i: i + nv])
            hyperedge_times_map[hyperedge].add(time)
            if hyperedge not in hyperedges:
                hyperedges.add(hyperedge)
                rows.extend(list(hyperedge))
                cols.extend([j] * len(hyperedge))
                hyperedge_list.append(hyperedge)
                j += 1
            # else:
            #     hyperedge_time_map[hyperedge] = min([hyperedge_time_map[hyperedge], time])
            i += nv
        m = len(hyperedges)
        print('Creating sparse matrix...')
        print(len(rows), len(cols), n, m)
        S = csr_matrix(([1] * len(rows), (rows, cols)), shape=(n, m))
        print('Preparing vertex list...')
        vertex_list = [Vertex(i, labels[i]) for i in tqdm(range(n))]

        print('Recalculating hyperedge times...')
        times = np.array([min(hyperedge_times_map[hyperedge_list[j]]) for j in tqdm(range(S.shape[1]))])

        id_label_map = {v.id: v.label for v in vertex_list}
        if split_mode == 'structural':
            min_size_limit = 2
            print('Filtering size for [{}, {}]'.format(min_size_limit, max_size_limit))
            S, times = filter_size(S, times, min_size_limit, max_size_limit)

        S, times, id_label_map = filter_time(S, times, id_label_map, st_time or -1, en_time or -1)

        if split_mode == 'structural':  # WARNING: DO NOT MOVE THIS FROM HERE; IT HAS TO BE THE LAST STEP
            times = np.array([0]*times.shape[0])
            print('WARNING: Time information is defaulted to all zeros (0) since structural mode')
        print('DATA STATS: S.shape = {}'.format(S.shape))
        return S, times, id_label_map
    
   
    


# @memory.cache
# def parse_benson_incidence_matrix(name,
#                                   base_path=None,
#                                   split_mode=None,
#                                   max_size_limit=None,
#                                   st_time=None,
#                                   en_time=None):
#     base_path = base_path or DATA_DEFAULTS['base_path_benson']
#     path = os.path.join(base_path, name)
#     nverts_path = os.path.join(path, name + '-nverts.txt')
#     simplices_path = os.path.join(path, name + '-simplices.txt')
#     times_path = os.path.join(path, name + '-times.txt')
#     print('Reading nverts...')
#     nverts = [int(l.rstrip('\n')) for l in tqdm(open(nverts_path, 'r'))]
#     # TODO: Remember that we are reindexing vertices to 0-index.
#     #  This has to be followed while initializing labels as well.
#     print('Reading simplices...')
#     simplices = [int(l.rstrip('\n')) - 1 for l in tqdm(open(simplices_path, 'r'))]
#     labels_path = os.path.join(path, name + '-node-labels.txt')
#     vertices = list(sorted(set(simplices)))
#     n = max(vertices) + 1
#     try:
#         print('Reading labels...')
#         labels = [l.rstrip('\n') for l in tqdm(open(labels_path, 'r'))]
#         labels = [' '.join(l.split(' ')[1:]) for l in labels]
#     except FileNotFoundError:
#         print('No labels found.')
#         labels = ['vertex_{}'.format(i) for i in range(n)]
#     print('Reading times...')
#     times = [float(l.rstrip('\n')) for l in tqdm(open(times_path, 'r'))]
#     hyperedges = set()
#     hyperedge_times_map = defaultdict(set)
#     hyperedge_list = []
#     rows = []
#     cols = []
#     j = 0
#     print('Parsing simplices...')
#     i = 0
#     iterator = list(zip(nverts, times))
#     for nv, time in tqdm(iterator):
#         hyperedge = frozenset(simplices[i: i + nv])
#         hyperedge_times_map[hyperedge].add(time)
#         if hyperedge not in hyperedges:
#             hyperedges.add(hyperedge)
#             rows.extend(list(hyperedge))
#             cols.extend([j] * len(hyperedge))
#             hyperedge_list.append(hyperedge)
#             j += 1
#         # else:
#         #     hyperedge_time_map[hyperedge] = min([hyperedge_time_map[hyperedge], time])
#         i += nv
#     m = len(hyperedges)
#     print('Creating sparse matrix...')
#     print(len(rows), len(cols), n, m)
#     S = csr_matrix(([1] * len(rows), (rows, cols)), shape=(n, m))
#     print('Preparing vertex list...')
#     vertex_list = [Vertex(i, labels[i]) for i in tqdm(range(n))]

#     print('Recalculating hyperedge times...')
#     times = np.array([min(hyperedge_times_map[hyperedge_list[j]]) for j in tqdm(range(S.shape[1]))])

#     id_label_map = {v.id: v.label for v in vertex_list}
#     if split_mode == 'structural':
#         min_size_limit = 2
#         print('Filtering size for [{}, {}]'.format(min_size_limit, max_size_limit))
#         S, times = filter_size(S, times, min_size_limit, max_size_limit)

#     S, times, id_label_map = filter_time(S, times, id_label_map, st_time or -1, en_time or -1)

#     if split_mode == 'structural':  # WARNING: DO NOT MOVE THIS FROM HERE; IT HAS TO BE THE LAST STEP
#         times = np.array([0]*times.shape[0])
#         print('WARNING: Time information is defaulted to all zeros (0) since structural mode')
#     print('DATA STATS: S.shape = {}'.format(S.shape))
#     return S, times, id_label_map


def main():
    S, times, id_label_map = parse_benson_incidence_matrix('email-Enron', ignore_time=False)
    print(S.shape)


if __name__ == '__main__':
    main()
