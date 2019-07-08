from src.incidence_matrix import IncidenceMatrix
from src.vertex import generate_vertex, Vertex


def test_init_empty():
    S = IncidenceMatrix()
    S.pprint()
    assert S.shape == (0, 0)


def test_init_row_vector():
    S = IncidenceMatrix([[1, 2, 3]])
    S.pprint()
    assert S.shape == (1, 3)


def test_init_column_vector():
    S = IncidenceMatrix([[1], [2], [3]])
    S.pprint()
    assert S.shape == (3, 1)


def test_init_matrix():
    S = IncidenceMatrix([[1, 2, 3], [4, 5, 6]])
    S.pprint()
    assert S.shape == (2, 3)


def test_init_row_vector_labeled():
    S = IncidenceMatrix([[1, 2, 3]], [Vertex(i, 'vertex{}'.format(i)) for i in range(1, 2)])
    S.pprint(True)
    assert S.shape == (1, 3)


def test_init_column_vector_labeled():
    S = IncidenceMatrix([[1], [2], [3]], [Vertex(i, 'vertex{}'.format(i)) for i in range(1, 4)])
    S.pprint(True)
    assert S.shape == (3, 1)


def test_init_matrix_labeled():
    S = IncidenceMatrix([[1, 2, 3], [4, 5, 6]], [Vertex(i, 'vertex{}'.format(i)) for i in range(1, 3)])
    S.pprint(True)
    assert S.shape == (2, 3)
