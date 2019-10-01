import numpy as np
import pandas as pd
from scipy.sparse import triu
import pickle
from collections import defaultdict
from tqdm import tqdm_notebook
import sys

from src.data_preparer import S_to_A, S_to_B, incidence_to_hyperedges

sys.path.append("/content/gdrive/My Drive/Colab Notebooks/libraries/")
sys.path.append("/content/gdrive/My Drive/Colab Notebooks/libraries/hynetworkx")
base_path = '/content/gdrive/My Drive/Colab Notebooks/data/'


# def hyper_jaccard_min(pairs, S_train):
#     return hyper_jaccard(pairs, S_train, 'min')

# def hyper_jaccard_max(pairs, S_train):
#     return hyper_jaccard(pairs, S_train, 'max')

# def hyper_jaccard_avg(pairs, S_train):
#     return hyper_jaccard(pairs, S_train, 'avg')

def get_avg(_list):
    return sum(_list) * 1.0 / len(_list) if len(_list) > 0 else 0


from scipy.sparse import identity as eye
from scipy.sparse.linalg import inv


def hyper_katz(pairs, S_train, aggregator='min', alpha=0, beta=0.01, max_power=5):
    print('Finding A and B from S')
    A = S_to_A(S_train)
    B = S_to_B(S_train)
    I_n = eye(A.shape[0])
    I_m = eye(B.shape[0])
    print('Iterating over matrix powers...')
    scores = {'Avg': defaultdict(float)}
    for k in tqdm_notebook(range(1, max_power + 1), 'Matrix power: '):
        K_alpha = (alpha * A) ** k
        K_beta = (beta * B) ** k
        S_beta = S_train * K_beta * S_train.T
        for u, v in tqdm_notebook(pairs, 'Test pair: '):
            w = K_alpha[u, v] + S_beta[u, v]
            scores['Avg'][(u, v)] += w
    #     K_alpha = inv(I_n - alpha * A)
    #     print('Inverting for K_beta')
    #     K_beta = inv(I_m - beta * B)
    #     print(K_alpha.shape, K_beta.shape, S_train.shape)
    #     K = K_alpha + S_train*K_beta*S_train.T
    #     scores = {'Avg': {(u, v): K[u, v] for u, v in tqdm_notebook(pairs, 'Test pair: ')}}
    return scores


def hyper_degree_product(pairs, S_train, aggregator='min'):
    scores = {'Avg': {}}
    for u, v in tqdm_notebook(pairs, 'Test pair: '):
        hyn_u = incidence_to_hyperedges(S_train[:, S_train[u, :].nonzero()[1]])
        hyn_v = incidence_to_hyperedges(S_train[:, S_train[v, :].nonzero()[1]])
        scores['Avg'][(u, v)] = len(hyn_u) * len(hyn_v)
    return scores


def hyper_product(pairs, S_train, aggregator='min'):
    #     aggregator = {'min': min, 'max': max, 'avg': get_avg}[aggregator]
    scores = {'Min': {}, 'Max': {}, 'Avg': {}}
    for u, v in tqdm_notebook(pairs, 'Test pair: '):
        hyn_u = incidence_to_hyperedges(S_train[:, S_train[u, :].nonzero()[1]])
        hyn_v = incidence_to_hyperedges(S_train[:, S_train[v, :].nonzero()[1]])
        hyper_scores = get_all_hp_scores(hyn_u, hyn_v)
        if len(hyper_scores) == 0:
            hyper_scores = [0]
        scores['Min'][(u, v)] = min(hyper_scores)
        scores['Max'][(u, v)] = max(hyper_scores)
        scores['Avg'][(u, v)] = get_avg(hyper_scores)
    return scores


def hyper_pearson(pairs, S_train, aggregator='min'):
    #     aggregator = {'min': min, 'max': max, 'avg': get_avg}[aggregator]
    scores = {'Min': {}, 'Max': {}, 'Avg': {}}
    for u, v in tqdm_notebook(pairs, 'Test pair: '):
        hyn_u = incidence_to_hyperedges(S_train[:, S_train[u, :].nonzero()[1]])
        hyn_v = incidence_to_hyperedges(S_train[:, S_train[v, :].nonzero()[1]])
        hyper_scores = get_all_hpear_scores(hyn_u, hyn_v, S_train.shape[0])
        if len(hyper_scores) == 0:
            hyper_scores = [0]
        scores['Min'][(u, v)] = min(hyper_scores)
        scores['Max'][(u, v)] = max(hyper_scores)
        scores['Avg'][(u, v)] = get_avg(hyper_scores)
    return scores


def hyper_common_neighbour(pairs, S_train, aggregator='min'):
    #     aggregator = {'min': min, 'max': max, 'avg': get_avg}[aggregator]
    scores = {'Min': {}, 'Max': {}, 'Avg': {}}
    for u, v in tqdm_notebook(pairs, 'Test pair: '):
        hyn_u = incidence_to_hyperedges(S_train[:, S_train[u, :].nonzero()[1]])
        hyn_v = incidence_to_hyperedges(S_train[:, S_train[v, :].nonzero()[1]])
        hyper_scores = get_all_hcn_scores(hyn_u, hyn_v)
        if len(hyper_scores) == 0:
            hyper_scores = [0]
        scores['Min'][(u, v)] = min(hyper_scores)
        scores['Max'][(u, v)] = max(hyper_scores)
        scores['Avg'][(u, v)] = get_avg(hyper_scores)
    return scores


def hyper_min_overlap(pairs, S_train, aggregator='min'):
    #     aggregator = {'min': min, 'max': max, 'avg': get_avg}[aggregator]
    scores = {'Min': {}, 'Max': {}, 'Avg': {}}
    for u, v in tqdm_notebook(pairs, 'Test pair: '):
        hyn_u = incidence_to_hyperedges(S_train[:, S_train[u, :].nonzero()[1]])
        hyn_v = incidence_to_hyperedges(S_train[:, S_train[v, :].nonzero()[1]])
        hyper_scores = get_all_mino_scores(hyn_u, hyn_v)
        if len(hyper_scores) == 0:
            hyper_scores = [0]
        scores['Min'][(u, v)] = min(hyper_scores)
        scores['Max'][(u, v)] = max(hyper_scores)
        scores['Avg'][(u, v)] = get_avg(hyper_scores)
    return scores


def hyper_max_overlap(pairs, S_train, aggregator='min'):
    #     aggregator = {'min': min, 'max': max, 'avg': get_avg}[aggregator]
    scores = {'Min': {}, 'Max': {}, 'Avg': {}}
    for u, v in tqdm_notebook(pairs, 'Test pair: '):
        hyn_u = incidence_to_hyperedges(S_train[:, S_train[u, :].nonzero()[1]])
        hyn_v = incidence_to_hyperedges(S_train[:, S_train[v, :].nonzero()[1]])
        hyper_scores = get_all_maxo_scores(hyn_u, hyn_v)
        if len(hyper_scores) == 0:
            hyper_scores = [0]
        scores['Min'][(u, v)] = min(hyper_scores)
        scores['Max'][(u, v)] = max(hyper_scores)
        scores['Avg'][(u, v)] = get_avg(hyper_scores)
    return scores


def hyper_jaccard(pairs, S_train, aggregator='min'):
    #     aggregator = {'min': min, 'max': max, 'avg': get_avg}[aggregator]
    scores = {'Min': {}, 'Max': {}, 'Avg': {}}
    for u, v in tqdm_notebook(pairs, 'Test pair: '):
        hyn_u = incidence_to_hyperedges(S_train[:, S_train[u, :].nonzero()[1]])
        hyn_v = incidence_to_hyperedges(S_train[:, S_train[v, :].nonzero()[1]])
        hyper_scores = get_all_jc_scores(hyn_u, hyn_v)
        if len(hyper_scores) == 0:
            hyper_scores = [0]
        scores['Min'][(u, v)] = min(hyper_scores)
        scores['Max'][(u, v)] = max(hyper_scores)
        scores['Avg'][(u, v)] = get_avg(hyper_scores)
    return scores


def hyper_association_strength(pairs, S_train, aggregator='min'):
    #     aggregator = {'min': min, 'max': max, 'avg': get_avg}[aggregator]
    scores = {'Min': {}, 'Max': {}, 'Avg': {}}
    for u, v in tqdm_notebook(pairs, 'Test pair: '):
        hyn_u = incidence_to_hyperedges(S_train[:, S_train[u, :].nonzero()[1]])
        hyn_v = incidence_to_hyperedges(S_train[:, S_train[v, :].nonzero()[1]])
        hyper_scores = get_all_as_scores(hyn_u, hyn_v)
        if len(hyper_scores) == 0:
            hyper_scores = [0]
        scores['Min'][(u, v)] = min(hyper_scores)
        scores['Max'][(u, v)] = max(hyper_scores)
        scores['Avg'][(u, v)] = get_avg(hyper_scores)
    return scores


def hyper_cosine(pairs, S_train, aggregator='min'):
    #     aggregator = {'min': min, 'max': max, 'avg': get_avg}[aggregator]
    scores = {'Min': {}, 'Max': {}, 'Avg': {}}
    for u, v in tqdm_notebook(pairs, 'Test pair: '):
        hyn_u = incidence_to_hyperedges(S_train[:, S_train[u, :].nonzero()[1]])
        hyn_v = incidence_to_hyperedges(S_train[:, S_train[v, :].nonzero()[1]])
        hyper_scores = get_all_cos_scores(hyn_u, hyn_v)
        if len(hyper_scores) == 0:
            hyper_scores = [0]
        scores['Min'][(u, v)] = min(hyper_scores)
        scores['Max'][(u, v)] = max(hyper_scores)
        scores['Avg'][(u, v)] = get_avg(hyper_scores)
    return scores


def hyper_n_measure(pairs, S_train, aggregator='min'):
    #     aggregator = {'min': min, 'max': max, 'avg': get_avg}[aggregator]
    scores = {'Min': {}, 'Max': {}, 'Avg': {}}
    for u, v in tqdm_notebook(pairs, 'Test pair: '):
        hyn_u = incidence_to_hyperedges(S_train[:, S_train[u, :].nonzero()[1]])
        hyn_v = incidence_to_hyperedges(S_train[:, S_train[v, :].nonzero()[1]])
        hyper_scores = get_all_n_measure_scores(hyn_u, hyn_v)
        if len(hyper_scores) == 0:
            hyper_scores = [0]
        scores['Min'][(u, v)] = min(hyper_scores)
        scores['Max'][(u, v)] = max(hyper_scores)
        scores['Avg'][(u, v)] = get_avg(hyper_scores)
    return scores


def get_all_hp_scores(hyn_u, hyn_v):
    hp_scores = []
    for hu in hyn_u:
        for hv in hyn_v:
            prod = len(hu) * len(hv)
            hp_scores.append(prod)

    return hp_scores


def get_all_hcn_scores(hyn_u, hyn_v):
    hcn_scores = []
    for hu in hyn_u:
        for hv in hyn_v:
            intersection_count = len(hu.intersection(hv))
            hcn_scores.append(intersection_count)

    return hcn_scores


def get_all_jc_scores(hyn_u, hyn_v):
    jc_scores = []
    for hu in hyn_u:
        for hv in hyn_v:
            intersection_count = len(hu.intersection(hv))
            union_count = len(hu.union(hv))
            jc = intersection_count * 1.0 / union_count
            jc_scores.append(jc)
    # print(hyn_u)
    # print(hyn_v)
    # print(jc_scores)
    return jc_scores


def get_all_mino_scores(hyn_u, hyn_v):
    mino_scores = []
    for hu in hyn_u:
        for hv in hyn_v:
            intersection_count = len(hu.intersection(hv))
            z = min(len(hu), len(hv))
            jc = intersection_count * 1.0 / z
            mino_scores.append(jc)
    return mino_scores


def get_all_maxo_scores(hyn_u, hyn_v):
    maxo_scores = []
    for hu in hyn_u:
        for hv in hyn_v:
            intersection_count = len(hu.intersection(hv))
            z = max(len(hu), len(hv))
            jc = intersection_count * 1.0 / z
            maxo_scores.append(jc)
    return maxo_scores


def get_all_as_scores(hyn_u, hyn_v):
    as_scores = []
    for hu in hyn_u:
        for hv in hyn_v:
            intersection_count = len(hu.intersection(hv))
            w = intersection_count / \
                (len(hu) * len(hv))
            as_scores.append(w)

    return as_scores


def get_all_cos_scores(hyn_u, hyn_v):
    cos_scores = []
    for hu in hyn_u:
        for hv in hyn_v:
            intersection_count = len(hu.intersection(hv))
            w = intersection_count / \
                np.sqrt(len(hu) * len(hv))
            cos_scores.append(w)

    return cos_scores


def get_all_n_measure_scores(hyn_u, hyn_v):
    nm_scores = []
    for hu in hyn_u:
        for hv in hyn_v:
            intersection_count = len(hu.intersection(hv))

            w = np.sqrt(2) * \
                intersection_count / \
                np.sqrt(len(hu) ** 2 + len(hv) ** 2)
            nm_scores.append(w)
    return nm_scores


# fix pearson

def get_all_hpear_scores(hyn_u, hyn_v, n):
    hpear_scores = []
    #     a=len(hyn_u)
    #     b=len(hyn_v)
    for hu in hyn_u:
        a = len(hu)
        for hv in hyn_v:
            b = len(hv)
            intersection_count = len(hu.intersection(hv))
            numerator = (n * intersection_count) - (a * b)
            denominator = np.sqrt(n * a - a ** 2) * \
                          np.sqrt(n * b - b ** 2)
            w = numerator / denominator
            if w > 0:
                hpear = w
            else:
                hpear = 0
            hpear_scores.append(hpear)
    return hpear_scores


# def get_all_jc_scores(S_u, S_v):
#     jc_scores = []
#     hd_u = S_u.shape[1]
#     hd_v = S_v.shape[1]
#     for i in range(hd_u):
#         S_u_i = S_u[:, i]
#         for j in range(hd_v):
#             S_v_j = S_v[:, j]
#             intersection_count = (S_u_i.multiply(S_v_j)).nnz
#             union_count = (S_u_i + S_v_j).nnz
#             jc = intersection_count * 1.0 / union_count
#             jc_scores.append(jc)
#     return jc_scores

# def hyper_adamic_adar_min(pairs, S_train):
#     return hyper_adamic_adar(pairs, S_train, 'min')

# def hyper_adamic_adar_max(pairs, S_train):
#     return hyper_adamic_adar(pairs, S_train, 'max')

# def hyper_adamic_adar_avg(pairs, S_train):
#     return hyper_adamic_adar(pairs, S_train, 'avg')

def hyper_adamic_adar(pairs, S_train, aggregator='min'):
    #     aggregator = {'min': min, 'max': max, 'avg': get_avg}[aggregator]
    scores = {'Min': {}, 'Max': {}, 'Avg': {}}
    hyd_array = S_train.sum(axis=1)
    for u, v in tqdm_notebook(pairs, 'Test pair: '):
        hyn_u = incidence_to_hyperedges(S_train[:, S_train[u, :].nonzero()[1]])
        hyn_v = incidence_to_hyperedges(S_train[:, S_train[v, :].nonzero()[1]])
        hyper_scores = get_all_aa_scores(hyn_u, hyn_v, hyd_array)
        if len(hyper_scores) == 0:
            hyper_scores = [0]
        scores['Min'][(u, v)] = min(hyper_scores)
        scores['Max'][(u, v)] = max(hyper_scores)
        scores['Avg'][(u, v)] = get_avg(hyper_scores)
    return scores


# def get_all_aa_scores(S, S_u, S_v):
#     aa_scores = []
#     hd_u = S_u.shape[1]
#     hd_v = S_v.shape[1]
#     for i in range(hd_u):
#         S_u_i = S_u[:, i]
#         for j in range(hd_v):
#             S_v_j = S_v[:, j]
#             common_nodes = (S_u_i.multiply(S_v_j)).nonzero()[0]
#             # TODO: Check the base of the log here and in linkpred AA
#             aa = sum((1/(1 + np.log(S[common_nodes, :].sum(axis=1)))).T.tolist()[0])
#             aa_scores.append(aa)
#     return aa_scores


def get_all_aa_scores(hyn_u, hyn_v, hyd_array):
    aa_scores = []
    for hu in hyn_u:
        for hv in hyn_v:
            common_nodes = hu.intersection(hv)
            # TODO: Check the base of the log here and in linkpred AA
            if len(common_nodes) > 0:
                #                 print(common_nodes)
                #                 print(S[common_nodes, :])
                aa = sum((1 / (1 + np.log(hyd_array[list(common_nodes)]))).T.tolist()[0])
            else:
                aa = 0
            aa_scores.append(aa)
    return aa_scores


all_hypergraph_score_names = ['HyperJaccardMin',  # 0
                              'HyperJaccardMax',  # 1
                              'HyperJaccardAvg',  # 2
                              'HyperAdamicAdarMin',  # 3
                              'HyperAdamicAdarMax',  # 4
                              'HyperAdamicAdarAvg',  # 5
                              'HyperDegreeProductAvg',  # 6
                              'HyperProductMin',  # 7
                              'HyperProductMax',  # 8
                              'HyperProductAvg',  # 9
                              'HyperPearsonMin',  # 10
                              'HyperPearsonMax',  # 11
                              'HyperPearsonAvg',  # 12
                              'HyperAssociationStrengthMin',  # 13
                              'HyperAssociationStrengthMax',  # 14
                              'HyperAssociationStrengthAvg',  # 15
                              'HyperCosineMin',  # 16
                              'HyperCosineMax',  # 17
                              'HyperCosineAvg',  # 18
                              'HyperNMeasureMin',  # 19
                              'HyperNMeasureMax',  # 20
                              'HyperNMeasureAvg',  # 21
                              'HyperKatzAvg',  # 22
                              'HyperCommonNeighbourMin',  #
                              'HyperCommonNeighbourMax',  #
                              'HyperCommonNeighbourAvg',
                              'HyperMinOverlapMin',  #
                              'HyperMinOverlapMax',  #
                              'HyperMinOverlapAvg',
                              'HyperMaxOverlapMin',  #
                              'HyperMaxOverlapMax',  #
                              'HyperMaxOverlapAvg',

                              ]

# all_predictor_names = ['Random','AdamicAdar', 'CommonNeighbours', 'Cosine', 'DegreeProduct', 'Jaccard', 'Katz', 'SimRank']
hypergraph_score_abbr_map = {'HyperJaccardMin': 'HJCm',
                             'HyperJaccardMax': 'HJCM',
                             'HyperJaccardAvg': 'HJCa',
                             'HyperAdamicAdarMin': 'HAAm',
                             'HyperAdamicAdarMax': 'HAAM',
                             'HyperAdamicAdarAvg': 'HAAa',
                             'HyperDegreeProductAvg': 'HDPa',
                             'HyperProductMin': 'HPm',
                             'HyperProductMax': 'HPM',
                             'HyperProductAvg': 'HPa',
                             'HyperPearsonMin': 'HPearm',
                             'HyperPearsonMax': 'HPearM',
                             'HyperPearsonAvg': 'HPeara',
                             'HyperAssociationStrengthMin': 'HASm',
                             'HyperAssociationStrengthMax': 'HASM',
                             'HyperAssociationStrengthAvg': 'HASa',
                             'HyperCosineMin': 'HCosm',
                             'HyperCosineMax': 'HCosM',
                             'HyperCosineAvg': 'HCosa',
                             'HyperNMeasureMin': 'HNMm',
                             'HyperNMeasureMax': 'HNMM',
                             'HyperNMeasureAvg': 'HNMa',
                             'HyperKatzAvg': 'HKz',
                             'HyperCommonNeighbourMin': 'HCNm',
                             'HyperCommonNeighbourMax': 'HCNM',
                             'HyperCommonNeighbourAvg': 'HCNa',
                             'HyperMinOverlapMin': 'Hminom',  #
                             'HyperMinOverlapMax': 'HminoM',  #
                             'HyperMinOverlapAvg': 'HminoA',
                             'HyperMaxOverlapMin': 'Hmaxom',  #
                             'HyperMaxOverlapMax': 'HmaxoM',  #
                             'HyperMaxOverlapAvg': 'HmaxoA',

                             }

scoring_function_map = {'HyperJaccard': hyper_jaccard,
                        'HyperAdamicAdar': hyper_adamic_adar,
                        'HyperDegreeProduct': hyper_degree_product,
                        'HyperProduct': hyper_product,
                        'HyperPearson': hyper_pearson,
                        'HyperAssociationStrength': hyper_association_strength,
                        'HyperCosine': hyper_cosine,
                        'HyperNMeasure': hyper_n_measure,
                        'HyperKatz': hyper_katz,
                        'HyperCommonNeighbour': hyper_common_neighbour,
                        'HyperMinOverlap': hyper_min_overlap,
                        'HyperMaxOverlap': hyper_max_overlap,

                        }


def get_hypergraph_scores(lp_data, score_indices=None):
    score_indices = score_indices or range(len(all_hypergraph_score_names))
    score_names = [all_hypergraph_score_names[i] for i in score_indices]
    base_score_names = list({n[:-3] for n in score_names})
    S_train = lp_data['S_train']
    A_test_pos = lp_data['A_test_pos']
    A_test_neg = lp_data['A_test_neg']
    I, J = triu(A_test_pos + A_test_neg).nonzero()
    test_pairs = list(zip(I, J))
    scores = {}
    for i in tqdm_notebook(range(len(base_score_names)), 'Hypergraph score: '):
        # print(len(base_score_names))
        base_name = base_score_names[i]
        # print(base_name)
        scoring_function = scoring_function_map[base_name]
        print('Calculating hypergraph scores: {}'.format(base_name))
        base_scores = scoring_function(test_pairs, S_train)
        for prefix in base_scores:
            score_name = base_name + prefix
            abbr = hypergraph_score_abbr_map[score_name]
            scores[abbr] = base_scores[prefix]
    scores_df = pd.DataFrame(scores)
    return scores_df


def store_hypergraph_scores(S_train, test_pairs, file_prefix, score_indices=None):
    score_indices = score_indices or range(len(all_hypergraph_score_names))
    score_names = [all_hypergraph_score_names[i] for i in score_indices]
    base_score_names = list({n[:-3] for n in score_names})
    for i in tqdm_notebook(range(len(base_score_names)), 'Hypergraph score: '):
        base_name = base_score_names[i]
        scoring_function = scoring_function_map[base_name]
        print('Calculating hypergraph scores: {}'.format(base_name))
        base_scores = scoring_function(test_pairs, S_train)
        for prefix in base_scores:
            score_name = base_name + prefix
            abbr = hypergraph_score_abbr_map[score_name]
            scores = [base_scores[prefix][pair] for pair in test_pairs]
            file_path = file_prefix + '_' + abbr
            print('Pickling into {}'.format(file_path))
            pickle.dump(scores, open(file_path, 'wb'))
            del scores
        del base_scores
