import random
import numpy as np
from scipy.sparse import csr_matrix, triu, hstack
from collections import defaultdict
from tqdm import tqdm
import sys

from src.utils import get_base_path

base_path = get_base_path()
default_time_filter_params = (None, None)
data_time_filter_params_map = {'coauth-DBLP': (1980, 1982),
                               'threads-math-sx': (206041827210, None),  # Took last 100000 posts
                               'tags-math-sx': (217219821220, None),  # Took last 100000 posts
                               }


def get_time_filter_params(data_name):
    if data_name in data_time_filter_params_map:
        return data_time_filter_params_map[data_name]
    return default_time_filter_params


def filter_size(X, times, min_size=0, max_size=-1):
    if max_size == -1:
        max_size = X.shape[0]
    x = X.sum(axis=0)
    _filter = ((x <= max_size) & (x >= min_size)).nonzero()[1]
    times = times[_filter]
    X = X[:, _filter]
    return X, times


def filter_time(S, times, id_label_map, min_time=-1, max_time=-1):
    if min_time == -1 and max_time == -1:
        return S, times, id_label_map
    if min_time == -1:
        min_time = min(times)
    if max_time == -1:
        max_time = max(times)
    print('Filtering time for [{}, {}]'.format(min_time, max_time))
    _filter = ((times <= max_time) & (times >= min_time)).nonzero()[0]
    S = S[:, _filter]
    times = times[_filter]
    V_filtered = list(sorted(S.sum(axis=1).nonzero()[0].tolist()))
    S = S[V_filtered, :]
    id_label_map = {i: id_label_map[v] for i, v in enumerate(V_filtered)}
    return S, times, id_label_map


def prepare_temporal_lp_data(S, weighted, times, rho, neg_ratio=-1, mode='random'):
    print('Generating A and edge-times...')
    A, edge_time_map = S_to_A_timed(S, weighted, times)
    print('Splitting train-test...')
    A_train, A_test, A_test_pos, train_end_time = split_train_test_temporal(A, weighted, edge_time_map, rho)
    print('Generating negative data')
    A_test_neg = get_neg_data(A, A_test_pos, neg_ratio, mode)
    S_train = S[:, [j for j, t in enumerate(times) if t <= train_end_time]]
    lp_data = {'S_train': S_train, 'A_train': A_train, 'A_test': A_test,
               'A_test_pos': A_test_pos, 'A_test_neg': A_test_neg}
    # assert ((lp_data['A_test_pos'] + lp_data['A_test_neg']) > 1).nnz == 0, "Negative test pairs overlap with positive."
    return lp_data


def prepare_lp_data(S, weighted, times, rho, neg_ratio=-1, mode='random'):
    if all([x == 0 for x in times]):
        print('Going for a structural split')
        return prepare_structural_lp_data(S, weighted, rho, neg_ratio, mode)
    else:
        print('Going for a temporal split')
        return prepare_temporal_lp_data(S, weighted, times, rho, neg_ratio, mode)


def prepare_structural_lp_data(S, weighted, rho, neg_ratio=-1, mode='random'):
    A = S_to_A(S, weighted, silent=False)
    print('Splitting into train/test...')
    A_train, A_test, A_test_pos = split_train_test(A, weighted, rho)
    A_test_neg = get_neg_data(A, A_test_pos, neg_ratio, mode)
    S_train = clean_train_hypergraph(S, A_test_pos)
    lp_data = {'S_train': S_train, 'A_train': A_train, 'A_test': A_test,
               'A_test_pos': A_test_pos, 'A_test_neg': A_test_neg}
    # assert ((lp_data['A_test_pos'] + lp_data['A_test_neg']) > 1).nnz == 0, "Negative test pairs overlap with positive."
    return lp_data


def S_to_A(S, weighted, silent=True):
    if not silent:
        print('Converting S to A')
        print('STEP 1: Computing SS^T')
    A = S * S.T

    if weighted == False:
        if not silent:
            print('STEP 2: Binarizing A')
        A[A > 0] = 1
    if not silent:
        print('STEP 3: Setting diagonals to 0')
    A.setdiag(0)
    if not silent:
        print('STEP 4: Eliminating zeros')
    A.eliminate_zeros()
    return A


def S_to_B(S, weighted=False):
    return S_to_A(S.T, weighted)


def S_to_A_timed(S, weighted, times):
    A = S_to_A(S, weighted)
    I, J = triu(A).nonzero()
    edges = list(zip(I, J))
    edge_time_map = {}
    node_hids_map = defaultdict(set)
    I, J = S.nonzero()
    iterator = list(zip(I, J))
    print('Precomputing node-hyperneighbor map...')
    for i, j in tqdm(iterator):
        node_hids_map[i].add(j)
    print('Resolving times for all edges...')
    for u, v in tqdm(edges):
        #         S_u = set(S[u, :].nonzero()[1])
        #         S_v = set(S[v, :].nonzero()[1])
        S_u = node_hids_map[u]
        S_v = node_hids_map[v]
        common_hyperedge_ids = S_u.intersection(S_v)
        edge_time = min([times[i] for i in common_hyperedge_ids])
        edge_time_map[(u, v)] = edge_time
    print(A.shape, len(edge_time_map))
    return A, edge_time_map


def split_train_test(A, weighted, rho):
    I, J = triu(A).nonzero()
    edges = set(zip(I, J))
    print('STEP 1: Sampling test edges...')
    test_edges = list(random.sample(edges, int(rho * len(edges))))
    test_I, test_J = list(zip(*test_edges))

    print('STEP 2: Preparing test data...')
    test_V = ([1] * len(test_I + test_J)) if not weighted else \
        ([A[i, j] for i, j in test_edges] + [A[j, i] for i, j in test_edges])
    A_test = csr_matrix((test_V, (test_I + test_J, test_J + test_I)), shape=A.shape)

    print('STEP 3: Preparing train data...')
    train_edges = list(edges.difference(set(test_edges)))
    train_I, train_J = list(zip(*train_edges))
    train_V = ([1] * len(train_I + train_J)) if not weighted else \
        ([A[i, j] for i, j in train_edges] + [A[j, i] for i, j in train_edges])
    A_train = csr_matrix((train_V, (train_I + train_J, train_J + train_I)), shape=A.shape)
    A_test_pos = A_test
    return A_train, A_test, A_test_pos


def split_train_test_temporal(A, weighted, edge_time_map, rho):
    timestamps = list(sorted(set(edge_time_map.values())))
    start_time, end_time = timestamps[0], timestamps[-1]
    duration = end_time - start_time
    train_duration = (1 - rho) * duration
    train_end_time = start_time + train_duration

    train_edges = [e for e in edge_time_map if edge_time_map[e] <= train_end_time]
    test_edges = [e for e in edge_time_map if edge_time_map[e] > train_end_time]

    if (weighted == False):
        test_I, test_J = zip(*test_edges)
        A_test = csr_matrix(([1] * len(test_I + test_J), (test_I + test_J, test_J + test_I)), shape=A.shape)
        train_I, train_J = zip(*train_edges)
        A_train = csr_matrix(([1] * len(train_I + train_J), (train_I + train_J, train_J + train_I)), shape=A.shape)
        assert ((A_train + A_test) != A).nnz == 0, "Error in train-test split"
        A_test_pos = A_test
        return A_train, A_test, A_test_pos, train_end_time

    if (weighted == True):
        test_I, test_J = zip(*test_edges)
        test_V = [A[i, j] for i, j in zip(test_I, test_J)] + [A[j, i] for i, j in zip(test_I, test_J)]
        A_test = csr_matrix((test_V, (test_I + test_J, test_J + test_I)), shape=A.shape)
        train_I, train_J = zip(*train_edges)
        train_V = [A[i, j] for i, j in zip(train_I, train_J)] + [A[j, i] for i, j in zip(train_I, train_J)]
        A_train = csr_matrix((train_V, (train_I + train_J, train_J + train_I)), shape=A.shape)
        assert ((A_train + A_test) != A).nnz == 0, "Error in train-test split"
        A_test_pos = A_test
        return A_train, A_test, A_test_pos, train_end_time


def remove_singleton_columns(X):
    x = X.sum(axis=0)
    X = X[:, (x > 1).nonzero()[1]]
    return X


def get_neg_data_optimized(A):
    A_neg = csr_matrix((A.shape[0], 0))
    print('Iterating over each row of adjacency matrix...')
    for i in tqdm(range(A.shape[0])):
        Ai = A[:, i]
        AAi = A * Ai
        AAi[AAi > 0] = 1
        AAi[Ai > 0] = 0
        AAi.eliminate_zeros()
        A_neg = hstack([A_neg, AAi])
    A_neg.setdiag(0)
    A_neg.eliminate_zeros()
    return A_neg


# def get_neg_data(A):
#     A_neg = A*A
#     A_neg[A_neg>0] = 1
#     A_neg[A>0] = 0
#     A_neg.setdiag(0)
#     A_neg.eliminate_zeros()
#     return A_neg


def get_random_pair(n):
    u = random.randrange(n)
    while True:
        v = random.randrange(n)
        if u != v:
            return (u, v)


def get_neg_data(A, A_pos, factor=-1, mode='random'):
    '''
    Caution: Use ratio not for a positive-negative ratio, but for the
    ratio of existing links to non-existing ones.
    '''
    if mode == 'neighbor':
        A_neg = A * A
        A_neg[A_neg > 0] = 1
        A_neg[A > 0] = 0
        A_neg.setdiag(0)
        A_neg.eliminate_zeros()
        if factor > 0:
            desired_neg_count = int(factor * triu(A_pos).nnz)
            neg_count = triu(A_neg).nnz
            if desired_neg_count > neg_count:
                print('Cannot generate {} negative samples; generating {}'.format(2 * desired_neg_count, 2 * neg_count))
            elif desired_neg_count < neg_count:
                neg_pairs = list(zip(*triu(A_neg).nonzero()))
                sampled_neg_pairs = random.sample(neg_pairs, int(desired_neg_count))
                I, J = zip(*sampled_neg_pairs)
                A_neg = csr_matrix(([1] * len(I + J), (I + J, J + I)), shape=A_neg.shape)
    elif mode == 'random':
        n = A.shape[0]
        edge_count = triu(A).nnz
        pos_count = triu(A_pos).nnz
        max_neg_count = (n ** 2 - n) / 2 - edge_count
        if factor > 0:
            desired_neg_count = int(factor * pos_count)
        else:
            desired_neg_count = max_neg_count + 1
        if max_neg_count < desired_neg_count:
            print('Cannot generate {} negative samples; generating {}'.format(2 * desired_neg_count, 2 * max_neg_count))
            A_neg = csr_matrix(A.shape, dtype=int)
            A_neg[A == 0] = 1
            A_neg.setdiag(0)
            A_neg.eliminate_zeros()
        else:
            neg_count = 0
            A_neg = csr_matrix(A.shape, dtype=int)
            neg_pairs = set()
            #             V = list(range(n))
            print('Generating negative patterns until {} are found'.format(desired_neg_count))
            pbar = tqdm(total=desired_neg_count)
            I, J = [], []
            while neg_count < desired_neg_count:
                u, v = get_random_pair(n)
                u, v = (u, v) if u <= v else (v, u)
                if not A[u, v] and (u, v) not in neg_pairs:
                    I.extend([u, v])
                    J.extend([v, u])
                    neg_pairs.add((u, v))
                    #                     A_neg[u, v] = 1
                    #                     A_neg[v, u] = 1
                    neg_count += 1
                    pbar.update(1)
            pbar.close()
            A_neg = csr_matrix(([1] * len(I), (I, J)), shape=A.shape)
    return A_neg


# def get_neg_data(A, A_pos, ratio=-1):
#     '''
#     Caution: Use ratio not for a positive-negative ratio, but for the
#     ratio of existing links to non-existing ones.
#     '''
#     A_neg = A*A
#     A_neg[A_neg>0] = 1
#     A_neg[A>0] = 0
#     A_neg.setdiag(0)
#     A_neg.eliminate_zeros()
#     if ratio > 0:
#         desired_neg_count = int(ratio * triu(A_pos).nnz)
#         neg_count = triu(A_neg).nnz
#         if desired_neg_count > neg_count:
#             print('Cannot generate {} negative samples; generating {}'.format(2*desired_neg_count, 2*neg_count))
#         elif desired_neg_count < neg_count:
#             neg_pairs = list(zip(*triu(A_neg).nonzero()))
#             sampled_neg_pairs = random.sample(neg_pairs, int(desired_neg_count))
#             I, J = zip(*sampled_neg_pairs)
#             A_neg = csr_matrix(([1]*len(I+J), (I+J, J+I)), shape=A_neg.shape)
#     return A_neg

def incidence_to_hyperedges(S, silent_mode=True, _type=set):
    I, J = S.nonzero()
    hyperedges = defaultdict(set)
    indices = list(zip(I, J))
    if not silent_mode:
        print('Converting incidence matrix to hyperedge {} for faster processing...'.format(_type))
    for i, j in (tqdm(indices) if not silent_mode else indices):
        hyperedges[j].add(i)
    if _type == set:
        return set(map(frozenset, hyperedges.values()))
    elif _type == list:
        return set(map(frozenset, hyperedges.values()))
    elif _type == dict:
        return {i: frozenset(f) for i, f in hyperedges.items()}
    return hyperedges


def hyperedges_to_incidence(hyperedges, nV):
    nF = len(hyperedges)
    hyperedges = list(set(hyperedges))
    I = []
    J = []
    for j, f in enumerate(hyperedges):
        I.extend(f)
        J.extend([j] * len(f))
    S = csr_matrix(([1] * len(I), (I, J)), shape=(nV, nF))
    return S


def clean_train_hypergraph(S, A_test_pos):
    I, J = triu(A_test_pos).nonzero()
    indices = list(zip(I, J))
    S_hyperedges = incidence_to_hyperedges(S, silent_mode=False)

    node_hnbrs_map = defaultdict(set)
    print('Precomputing node-hyperneighbor map...')
    for f in tqdm(S_hyperedges):
        for v in f:
            node_hnbrs_map[v].add(f)

    print('Splitting hyperedges and getting S_train...')
    for i, j in tqdm(indices):
        i_hnbrs = node_hnbrs_map[i]
        j_hnbrs = node_hnbrs_map[j]
        common_hyperedges = i_hnbrs.intersection(j_hnbrs)
        if len(common_hyperedges) == 0:
            continue
        for f in common_hyperedges:
            f_i = frozenset(f.difference({i}))
            f_j = frozenset(f.difference({j}))
            S_hyperedges.remove(f)
            S_hyperedges.add(f_i)
            S_hyperedges.add(f_j)
            for k in f:
                node_hnbrs_map[k].remove(f)
                node_hnbrs_map[k].add(f_i)
                node_hnbrs_map[k].add(f_j)
            node_hnbrs_map[i].remove(f_i)
            node_hnbrs_map[j].remove(f_j)
        # T = S[:, common_hyp_ids]
        # U = T.copy()
        # T[i, :] = 0
        # U[j, :] = 0
        # # TODO: check for repeated columns in S
        # V = hstack([T, U]).tocsr().astype(dtype=int)
        # V = remove_singleton_columns(V)
        # S = dropcols_fancy(S, common_hyp_ids)
        # V_hyperedges = incidence_to_hyperedges(V)
        # new_hyperedges = V_hyperedges.difference(S_hyperedges)
        # V_new = hyperedges_to_incidence(new_hyperedges, V.shape[0])
        # S = hstack([S, V_new]).tocsr().astype(dtype=int)
    S = hyperedges_to_incidence(S_hyperedges, S.shape[0])
    return S


def dropcols_fancy(M, idx_to_drop):
    idx_to_drop = np.unique(idx_to_drop)
    keep = ~np.in1d(np.arange(M.shape[1]), idx_to_drop, assume_unique=True)
    return M[:, np.where(keep)[0]]


def get_dummy_hypergraph():
    I = [1, 5, 1, 5, 8, 2, 5, 6, 7, 1, 3, 5, 2, 6, 7, 8, 1, 5, 4, 1]
    J = [1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 7]
    #     I = [1, 5, 1, 5, 8]
    #     J = [1, 1, 2, 2, 2]
    I = [i - 1 for i in I]
    J = [j - 1 for j in J]
    V = [1] * len(I)
    S = csr_matrix((V, (I, J)))
    return S
