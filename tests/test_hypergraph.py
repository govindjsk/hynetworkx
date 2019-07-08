from src.hypergraph import Hypergraph, generate_hypergraph
from src.vertex_set import generate_monotonic_vertex_set


def test_init_empty_hypergraph():
    H = Hypergraph()
    print(H.__summary__)
    assert len(H.V) == 0 and len(H.F) == 0


def test_init_semiempty_hypergraph():
    size = 5
    V = generate_monotonic_vertex_set(size)
    H = Hypergraph(V)
    print(H.__summary__)
    assert len(H.V) == size and len(H.F) == 0


def test_init_nonempty_hypergraph():
    num_vertices = 5
    num_hyperedges = 10
    H = generate_hypergraph(num_vertices, num_hyperedges)
    print(H.__summary__)
    assert len(H.V) == num_vertices and len(H.F) == num_hyperedges


def test_incidence_matrix():
    H = generate_hypergraph(10, 10)
    print(H.__summary__)
    H.S.pprint(True)
    assert H.S.shape == (10, 10)


def test_adjacency_matrix():
    H = generate_hypergraph(5, 2)
    print(H.__summary__)
    H.A.pprint(True)
    assert H.A.shape == (5, 5)
