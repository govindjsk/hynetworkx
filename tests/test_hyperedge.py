import pytest

from src.hyperedge import Hyperedge
from src.vertex import Vertex
from src.vertex_set import VertexSet


class TestCases:
    def __init__(self):
        self.ids = [[], [0], [0, 1], [0, 1, 2]]
        self.labels = [['vertex{}'.format(i) for i in ids] for ids in self.ids]
        self.vertices = [[Vertex(i, l) for i, l in zip(ids, labels)]
                         for ids, labels in zip(self.ids, self.labels)]
        self.vertices_V = [VertexSet(vertices) for vertices in self.vertices]

        self.ids_f = [Hyperedge(ids) for ids in self.ids]
        self.labels_f = [Hyperedge(labels) for labels in self.labels]
        self.vertices_f = [Hyperedge(vertices) for vertices in self.vertices]
        self.vertex_set_f = [Hyperedge(V) for V in self.vertices_V]


T = TestCases()


class TestInit:
    @pytest.mark.parametrize('actual, expected',
                             [(T.ids_f[i], set(T.vertices[i])) for i in range(4)])
    def test_init_with_ids(self, actual, expected):
        print(actual)
        assert {v.id for v in actual} == {v.id for v in expected}

    @pytest.mark.parametrize('actual, expected',
                             [(T.labels_f[i], set(T.vertices[i])) for i in range(4)])
    def test_init_with_labels(self, actual, expected):
        print(actual)
        assert actual == expected

    @pytest.mark.parametrize('actual, expected',
                             [(T.vertices_f[i], set(T.vertices[i])) for i in range(4)])
    def test_init_with_vertices(self, actual, expected):
        print(actual)
        assert actual == expected

    @pytest.mark.parametrize('actual, expected',
                             [(T.vertices_f[i], i) for i in range(4)])
    def test_cardinality(self, actual, expected):
        print(actual)
        assert len(actual) == expected

    def test_exception_on_non_iterable(self):
        with pytest.raises(Exception):
            Hyperedge(1)

    def test_exception_on_incompatible_iterable(self):
        with pytest.raises(Exception):
            Hyperedge([1.1, 1.2, 1.3])


def test_deduplicate_vertices_same():
    f = Hyperedge([Vertex(0, 'node0'), Vertex(0, 'node0')])
    print(f)
    assert len(f) == 1


def test_deduplicate_vertices_different():
    node = Vertex(0, 'node0')
    vertex = Vertex(0, 'vertex0')
    point = Vertex(0, 'point0')
    f_n = Hyperedge([node])
    f_p = Hyperedge([point])
    f_nv = Hyperedge([node, vertex])
    f_pn = Hyperedge([point, node])
    f_vp = Hyperedge([vertex, point])
    f_pnv = Hyperedge([point, vertex, node])
    print(f_nv, f_pn, f_vp, f_pnv)
    assert all([f_n == f_nv,
                f_n == f_pn,
                f_p == f_vp,
                f_n == f_pnv])


def test_hashability():
    f = T.vertices_f[0]
    print(hash(f))


def test_settability():
    f1 = T.vertices_f[1]
    f2 = T.vertices_f[1]
    F = {f1, Hyperedge([Vertex(0, 'node0')])}
    print(F)
    assert len(F) == 1, 'TODO: Have a separate class for HyperedgeSet and make it id-specific'
