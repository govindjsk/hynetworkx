from .utils import print_matrix, _or
from .vertex import Vertex
from scipy.sparse import csr_matrix


class AdjacencyMatrix(csr_matrix):
    def __init__(self, matrix=None, vertex_list=None):
        matrix = _or(matrix, csr_matrix, ([],))
        self.vertex_list = vertex_list
        super(AdjacencyMatrix, self).__init__(matrix)

    def pprint(self, show_labels=False):
        if show_labels:
            vertex_list = self.vertex_list or [Vertex(i) for i in range(self.shape[0])]
            row_labels = [v.id for v in vertex_list]
            print_matrix(self, row_labels)
        else:
            print_matrix(self)