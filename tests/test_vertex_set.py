from random import randrange

import pytest
from memory_profiler import profile

from src.vertex import Vertex, VERTEX_DEFAULTS
from src.vertex_set import VertexSet, VERTEX_SET_DEFAULTS, generate_vertex_set
from tests.test_vertex import VertexGenerationParameters


class TestCases:
    def __init__(self):
        self.ids = [[], [0], [0, 1], [0, 1, 2]]
        self.labels = [['vertex{}'.format(i) for i in ids] for ids in self.ids]
        self.vertices = [[Vertex(i, l) for i, l in zip(ids, labels)]
                         for ids, labels in zip(self.ids, self.labels)]
        self.ids_V = [VertexSet(ids) for ids in self.ids]
        self.labels_V = [VertexSet(labels) for labels in self.labels]
        self.vertices_V = [VertexSet(vertices) for vertices in self.vertices]


T = TestCases()


class TestInit:
    @pytest.mark.parametrize('actual, expected',
                             [(T.ids_V[i], set(T.vertices[i])) for i in range(4)])
    def test_init_with_ids(self, actual, expected):
        print(actual)
        assert {v.id for v in actual} == {v.id for v in expected}

    @pytest.mark.parametrize('actual, expected',
                             [(T.labels_V[i], set(T.vertices[i])) for i in range(4)])
    def test_init_with_labels(self, actual, expected):
        print(actual)
        assert actual == expected

    @pytest.mark.parametrize('actual, expected',
                             [(T.vertices_V[i], set(T.vertices[i])) for i in range(4)])
    def test_init_with_vertices(self, actual, expected):
        print(actual)
        assert actual == expected

    @pytest.mark.parametrize('actual, expected',
                             [(T.vertices_V[i], i) for i in range(4)])
    def test_cardinality(self, actual, expected):
        print(actual)
        assert len(actual) == expected

    def test_exception_on_non_iterable(self):
        with pytest.raises(Exception):
            VertexSet(1)

    def test_exception_on_incompatible_iterable(self):
        with pytest.raises(Exception):
            VertexSet([1.1, 1.2, 1.3])


class TestSetProperties:
    # TODO: Test for all elements, not only the first.
    @pytest.mark.parametrize('v, V', [(T.vertices[i][0], T.vertices_V[i]) for i in range(1, 4)])
    def test_containment(self, v, V):
        print("Testing {} in {}".format(v, V))
        assert v in V


class VertexSetGenerationParameters:
    def __init__(self):
        self.P = VertexGenerationParameters()
        self.parameters = []
        for param in self.P.parameters:
            count = randrange(0, 10)
            param = tuple([count] + list(param))
            self.parameters += [param]


G = VertexSetGenerationParameters()


class TestGenerateVertexSet:
    @profile
    @pytest.mark.parametrize('count, id_range, label_prefix, reserved_ids', G.parameters)
    def test_generate_vertex(self, count, id_range, label_prefix, reserved_ids):
        print(count, id_range, label_prefix, reserved_ids)
        resolved_count = count or VERTEX_SET_DEFAULTS['count']
        resolved_id_range = id_range or VERTEX_DEFAULTS['id_range']
        resolved_label_prefix = label_prefix or VERTEX_DEFAULTS['label_prefix']
        resolved_reserved_ids = set(reserved_ids or [])
        exception_expected = len(set(range(*resolved_id_range)).difference(resolved_reserved_ids)) < count
        if exception_expected:
            print('Exception expected.')
            with pytest.raises(Exception):
                generate_vertex_set(count, id_range, label_prefix, reserved_ids)
        else:
            V = generate_vertex_set(count, id_range, label_prefix, reserved_ids)
            print(V)
            assert all([V.__class__ == VertexSet,
                        len(V) == resolved_count,
                        all([resolved_id_range[0] <= v.id < resolved_id_range[1]] for v in V),
                        all([v.label == resolved_label_prefix + str(v.id)] for v in V),
                        all([v.id not in resolved_reserved_ids for v in V])])
