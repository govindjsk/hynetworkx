from src.hypergraph import Hypergraph, generate_hypergraph, parse_benson_hypergraph, Graph, generate_graph
from src.vertex_set import generate_monotonic_vertex_set
import pytest


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


@pytest.mark.parametrize('name, nV, nF', [
                                          ('contact-high-school', 327, 489),
                                          ('contact-primary-school', 242, 434),
                                          ('email-Enron', 148, 114)])
def test_parse_benson_hypergraph(name, nV, nF):
    H = parse_benson_hypergraph(name)
    print(H.S.shape)
    assert max(H.V)[0] + 1 == nV and len(H.F) == nF and \
        H.S.shape == (nV, nF) and H.A.shape == (nV, nV)


def test_graph_empty():
    G = Graph()
    print(G.__summary__)
    print(G)
    assert len(G.V) == 0 and len(G.E) == 0


def test_init_semiempty_graph():
    size = 5
    V = generate_monotonic_vertex_set(size)
    G = Graph(V)
    print(G.__summary__)
    assert len(G.V) == size and len(G.E) == 0


def test_init_nonempty_graph():
    num_vertices = 5
    num_edges = 10
    G = generate_graph(num_vertices, num_edges)
    print(G.__summary__)
    print(G)
    assert len(G.V) == num_vertices and len(G.E) == num_edges


def test_induce_graph():
    H = generate_hypergraph(5, 2, (3, 5))
    print(H.__summary__)
    assert not H._G_computed
    G = H.G
    assert H._G_computed
    print(G.__summary__)
    assert isinstance(G, Graph)
    assert G.V == H.V and all([any([e.issubset(f) for f in H.F]) for e in G.E]) and H.E == G.E


@pytest.mark.parametrize('name, nV, nF', [
                                        ('contact-high-school', 327, 489),
                                        ('contact-primary-school', 242, 434),
                                        ('email-Enron', 148, 114),
                                         ])
def test_parse_benson_hypergraph_graph(name, nV, nF):
    H = parse_benson_hypergraph(name)
    G = H.G
    print(G)
    assert G.V == H.V and H.E == G.E


def test_induced_graph_nx():
    H = generate_hypergraph(5, 2, (3, 5))
    print(H.__summary__)
    G = H.G.to_nx()
