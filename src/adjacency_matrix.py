from utils import print_matrix, _or, get_printable_matrix
from vertex import Vertex
from scipy.sparse import csr_matrix


class AdjacencyMatrix(csr_matrix):
    def __init__(self, matrix=None, vertex_list=None, **kwargs):
        matrix = _or(matrix, csr_matrix, ([],))
        self.vertex_list = vertex_list
        super(AdjacencyMatrix, self).__init__(matrix, **kwargs)

    def pprint(self, show_labels=False):
        if show_labels:
            vertex_list = self.vertex_list or [Vertex(i) for i in range(self.shape[0])]
            row_labels = [v.label for v in vertex_list]
            print_matrix(self, row_labels)
        else:
            print_matrix(self)

    def __repr__(self):
        return 'A {} x {} adjacency matrix with nnz = {}\n'.format(*self.shape, self.nnz) + \
               get_printable_matrix(self[:5, :5], self.vertex_list[:5])
