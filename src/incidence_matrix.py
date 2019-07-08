import numpy as np
from src.utils import print_matrix
from src.vertex import Vertex


class IncidenceMatrix(np.ndarray):
    def __new__(cls, matrix=None, vertex_list=None):
        matrix = matrix or np.array([])
        obj = np.asarray(matrix).view(cls)
        obj.vertex_list = vertex_list
        if len(obj.shape) == 1:
            if obj.shape[0] == 0:
                obj = obj.reshape(0, 0)
            else:
                obj = obj.reshape(obj.shape[0], 1)
        return obj

    # def __init__(self, matrix, vertex_list):
    #     super(IncidenceMatrix, self).__init__(matrix)

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.vertex_list = getattr(obj, 'vertex_list', None)

    def pprint(self, show_labels=False):
        if show_labels:
            vertex_list = self.vertex_list or [Vertex(i) for i in range(self.shape[0])]
            row_labels = [v.id for v in vertex_list]
            print_matrix(self, row_labels)
        else:
            print_matrix(self)
