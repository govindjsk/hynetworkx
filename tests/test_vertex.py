from itertools import product

import pytest
from memory_profiler import profile as profile

from src.vertex import Vertex, generate_vertex, VERTEX_DEFAULTS


class TestCases:
    def __init__(self):
        self.u = Vertex()
        self.v = Vertex()
        self.u0 = Vertex(0)
        self.v0 = Vertex(0)
        self.u0l = Vertex(0, 'vertex0')
        self.v0l = Vertex(0, 'vertex0')
        self.u1 = Vertex(1)
        self.v1 = Vertex(1)
        self.u1l = Vertex(1, 'vertex1')
        self.v1l = Vertex(1, 'vertex1')
        self.v1l_ = Vertex(1, 'node1')


T = TestCases()


class TestInit:
    def test_init(self):
        assert T.v.id == 0 and T.v.label is None

    def test_init_0(self):
        assert T.v0.id == 0 and T.v0.label is None

    def test_init_1(self):
        assert T.v1.id == 1 and T.v1.label is None

    def test_init_1l(self):
        assert T.v1l.id == 1 and T.v1l.label == 'vertex1'


class TestRepr:
    def test_repr(self):
        assert repr(T.v) == "Vertex(0, None)"

    def test_repr_0(self):
        assert repr(T.v0) == "Vertex(0, None)"

    def test_repr_0l(self):
        assert repr(T.v0l) == "Vertex(0, vertex0)"

    def test_repr_1l(self):
        assert repr(T.v1l) == "Vertex(1, vertex1)"

    def test_repr_1l_(self):
        assert repr(T.v1l_) == "Vertex(1, node1)"


class TestEquality:
    def test_eq(self):
        assert T.u == T.v

    def test_eq_0(self):
        assert T.u0 == T.v0

    def test_eq_1(self):
        assert T.u1 == T.v1

    def test_eq_1l(self):
        assert T.u1l == T.v1l

    def test_neq_11l(self):
        assert T.u1 != T.v1l

    def test_neq_1l1l_(self):
        assert T.v1l != T.v1l_

    def test_neq_01(self):
        assert T.v0 != T.v1

    def test_neq_01l(self):
        assert T.v0 != T.v1l


class TestCompare:
    def test_le_eq(self):
        assert T.u0l <= T.v0l

    def test_le_neq(self):
        assert T.v0l <= T.v1l

    def test_lt(self):
        assert T.v0l < T.v1l

    def test_ge_eq(self):
        assert T.u0l >= T.v0l

    def test_ge_neq(self):
        assert T.v1l >= T.v0l

    def test_gt(self):
        assert T.v1l > T.v0l


class TestHash:
    # def test_hash(self):
    #     h = hash(T.v)
    #     assert h == T.v.id

    def test_set(self):
        vertices = {T.v}
        print(vertices)
        assert T.v in vertices and len(vertices) == 1

    def test_set_duplicate(self):
        vertices = {T.u, T.v}
        print(vertices)
        assert T.u in vertices and T.v in vertices and len(vertices) == 1

    def test_set_doubleton(self):
        vertices = {T.v, T.v1}
        print(vertices)
        assert T.v in vertices and T.v1 in vertices and len(vertices) == 2


class VertexGenerationParameters:
    def __init__(self):
        self.id_ranges = [None, (0, 10), (5, 10)]
        self.label_prefixes = [None, 'node']
        self.reserved_ids = [None, range(100), range(10), range(5, 10), {6, 7}]
        self.parameters = product(self.id_ranges, self.label_prefixes, self.reserved_ids)


G = VertexGenerationParameters()


class TestGenerateVertex:
    @profile
    @pytest.mark.parametrize('id_range, label_prefix, reserved_ids', G.parameters)
    def test_generate_vertex(self, id_range, label_prefix, reserved_ids):
        print(id_range, label_prefix, reserved_ids)
        resolved_id_range = id_range or VERTEX_DEFAULTS['id_range']
        resolved_label_prefix = label_prefix or VERTEX_DEFAULTS['label_prefix']
        resolved_reserved_ids = set(reserved_ids or [])
        exception_expected = len(set(range(*resolved_id_range)).difference(resolved_reserved_ids)) == 0
        if exception_expected:
            print('Exception expected.')
            with pytest.raises(Exception):
                generate_vertex(id_range, label_prefix, reserved_ids)
        else:
            v = generate_vertex(id_range, label_prefix, reserved_ids)
            print(v)
            assert all([v.__class__ == Vertex,
                        resolved_id_range[0] <= v.id < resolved_id_range[1],
                        v.label == resolved_label_prefix + str(v.id),
                        v.id not in resolved_reserved_ids])
