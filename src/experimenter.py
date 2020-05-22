import multiprocessing
from concurrent.futures import as_completed, ProcessPoolExecutor
from random import sample
import argparse
import sys
import os
from collections import defaultdict
from pprint import pprint

from tqdm.autonotebook import tqdm

# sys.path.append('../')
import pandas as pd
# from scipy.stats import kendalltau
import numpy as np
from utils import get_library_path
library_path = get_library_path()
sys.path.append(library_path)
sys.path.append(os.path.join(library_path, "hynetworkx"))

from src.utils import get_data_abbr, mkdir_p, get_base_path, get_library_path
from src.data_preparer import filter_size, prepare_lp_data, get_time_filter_params, incidence_to_hyperedges, \
    hyperedges_to_incidence
from src.hypergraph_link_predictor import get_hypergraph_scores, hypergraph_score_abbr_map, all_hypergraph_score_names
from src.link_predictor import get_perf_df
from src.linkpred_predictor import get_linkpred_scores, predictor_abbr_map, all_predictor_names
from src.supervised_link_predictor import classify

from src.incidence_matrix import parse_benson_incidence_matrix as parse_S

from joblib import Memory

base_path = get_base_path()

defaults = {'data_name': 'email-Enron',
            'base_path': base_path,
            'split_mode': 'structural'}

cachedir = os.path.join(base_path, 'cache')

memory = Memory(cachedir, verbose=0)

data_params = {'data_name': 'email-Enron',
               'base_path': base_path,
               'split_mode': 'structural',
               'max_size_limit': 10}
lp_data_params = {'rho': 0.2,
                  'neg_factor': 5,
                  'neg_mode': 'random',
                  'weighted_flag': False}
lp_params = {'linkpred_indices': None,  # Say None for all scores
             'hypergraph_score_indices': None,  # Say None for all scores
             }
classifier_params = {'features': None,
                     'classifier': 'xgboost'}
column_s = ['classifier']
data_names = ['email-Enron', 'contact-high-school']
split_modes = ['structural', 'temporal']
weighted_flags = [False, True]

default_lp_cols = ['AA', 'AS', 'CN', 'Cos', 'PA', 'JC', 'MxO', 'MnO', 'NM', 'Prn']
abbr_pred_map = {a: p for p, a in predictor_abbr_map.items()}
default_lp_names = [abbr_pred_map[a] for a in default_lp_cols]
default_lp_indices = [all_predictor_names.index(p) for p in default_lp_names]

metrics = ['ap', 'auc', 'p@+', 'p@10', 'p@100', 'r@+', 'r@10', 'r@100']
default_hyper_cols = ['HAAM', 'HAAa', 'HAAl1', 'HAAl2',
                      'HASM', 'HASa', 'HASl1', 'HASl2',
                      'HCNM', 'HCNa', 'HCNl1', 'HCNl2',
                      'HCosM', 'HCosa', 'HCosl1', 'HCosl2',
                      'HDPa', 'HPM', 'HPa', 'HPl1', 'HPl2',
                      'HJCM', 'HJCa', 'HJCl1', 'HJCl2',
                      'HmaxoM', 'HmaxoA', 'Hmaxol1', 'Hmaxol2',
                      'HminoM', 'HminoA', 'Hminol1', 'Hminol2',
                      'HNMM', 'HNMa', 'HNMl1', 'HNMl2',
                      'HPearM', 'HPeara', 'HPearl1', 'HPearl2',
                      ]
hyg_abbr_pred_map = {a: p for p, a in hypergraph_score_abbr_map.items()}
default_hyg_names = [hyg_abbr_pred_map[a] for a in default_hyper_cols]
default_hyper_indices = [all_hypergraph_score_names.index(p) for p in default_hyg_names]

combined_tables = {}
mixed_combinations_map = {'AA': ['AA', 'HAAM', 'HAAa', 'HAAl1', 'HAAl2'],
                          'AS': ['AS', 'HASM', 'HASa', 'HASl1', 'HASl2'],
                          'CN': ['CN', 'HCNM', 'HCNa', 'HCNl1', 'HCNl2'],
                          'Cos': ['Cos', 'HCosM', 'HCosa', 'HCosl1', 'HCosl2'],
                          'PA': ['PA', 'HDPa', 'HPM', 'HPa', 'HPl1', 'HPl2'],
                          'JC': ['JC', 'HJCM', 'HJCa', 'HJCl1', 'HJCl2'],
                          'Kz': ['Kz', 'HKz'],
                          'MxO': ['MxO', 'HmaxoM', 'HmaxoA', 'Hmaxol1', 'Hmaxol2'],
                          'MnO': ['MnO', 'HminoM', 'HminoA', 'Hminol1', 'Hminol2'],
                          'NM': ['NM', 'HNMM', 'HNMa', 'HNMl1', 'HNMl2'],
                          'Prn': ['Prn', 'HPearM', 'HPeara', 'HPearl1', 'HPearl2'], }


# @memory.cache
def perform_classification(data_params, lp_data_params, lp_params, classifier_params, iter_var=0):
    _, _, lp_results = perform_link_prediction(data_params, lp_data_params, lp_params, iter_var)
    features, classifier = [classifier_params[x] for x in ['features', 'classifier']]
    classifier_output = classify(lp_results, features, classifier, iter_var)
    return classifier_output


def populate_and_store_classifier_tables(data_names, split_modes, feature_combinations_map, metrics, classifier, path,
                                         random_state=42, iter_var=0):
    if classifier == 'decision_tree':
        print("not handled")
        return 0
    classifier_params['classifier'] = classifier
    tables = {}
    for split_mode in split_modes:
        tables[split_mode] = {comb_name: defaultdict(list) for comb_name in feature_combinations_map}
        data_params['split_mode'] = split_mode
        for data_name in data_names:
            data_params['data_name'] = data_name
            for comb_name, comb in feature_combinations_map.items():
                #                 tables[split_mode][comb_name] = defaultdict(list)
                classifier_params['features'] = comb
                perf_df, _, _ = perform_classification(data_params, lp_data_params, lp_params, classifier_params,
                                                       random_state, iter_var)
                for metric in metrics:
                    row = perf_df.loc[metric, [classifier]]
                    row.name = get_data_abbr(data_name)
                    row.rename(index={classifier: '${}_{{{}}}$'.format(*comb_name)}, inplace=True)
                    tables[split_mode][comb_name][metric].append(row)

    updated_tables = {}
    for split_mode in split_modes:
        updated_tables[split_mode] = {}
        for metric in metrics:
            dfs = []
            for comb_name, comb in feature_combinations_map.items():
                df = pd.DataFrame(tables[split_mode][comb_name][metric]).T
                dfs.append(df)
            df = pd.concat(dfs)
            df = df.rename_axis(metric, axis=1)
            df = df.rename(columns={c: '\\textbf{{{}}}'.format(c) for c in df.columns})
            updated_tables[split_mode][metric] = df
            mkdir_p(os.path.join(base_path, 'tables', path))
            file_path = os.path.join(base_path, 'tables', path, '_'.join([split_mode, metric]) + '.tex')
            df.to_latex(open(file_path, 'w'), bold_rows=True, escape=False)
            # print(df.to_latex(bold_rows=True, escape=False))
    return updated_tables


# @memory.cache
def perform_link_prediction(data_params, lp_data_params, lp_params=None, ter_var=0, include_train=False, data=None, parallel_version=False):
    """
    data_params: {'data_name', 'base_path', 'split_mode', 'max_size_limit'}
    lp_data_params: {'rho', 'neg_factor', 'neg_mode', 'weighted_flag'}
    lp_params: {'linkpred_indices', 'hypergraph_score_indices'}

    returns: (data, lp_data, lp_results)
    """
    if data is None:
        print('READING DATASET...')
        data_name, base_path, split_mode, max_size_limit = [data_params[x] for x in
                                                            ['data_name', 'base_path', 'split_mode', 'max_size_limit']]
        S, times, id_label_map = parse_S(data_name,
                                         base_path,
                                         split_mode,
                                         max_size_limit,
                                         *get_time_filter_params(data_name))
        data = (S, times, id_label_map)
    else:
        S, times, id_label_map = data

    # print('PREPARING LP DATA...')
    rho, neg_factor, neg_mode = [lp_data_params[x] for x in
                                 ['rho', 'neg_factor', 'neg_mode']]
    weighted_lp_data = prepare_lp_data(S, True, times, rho, neg_factor, neg_mode)

    # print('PERFORMING LINK PREDICTION...')
    if lp_params:
        linkpred_indices, hypergraph_score_indices = [lp_params[x] for x in
                                                      ['linkpred_indices', 'hypergraph_score_indices']]
    else:
        linkpred_indices, hypergraph_score_indices = None, None
    weighted_linkpred_scores_df = get_linkpred_scores(weighted_lp_data, True, linkpred_indices,
                                                      include_train=include_train, parallel_version=parallel_version)
    unweighted_linkpred_scores_df = get_linkpred_scores(weighted_lp_data, False, linkpred_indices,
                                                        include_train=include_train, parallel_version=parallel_version)
    unweighted_linkpred_cols = list(unweighted_linkpred_scores_df.columns)
    cols_map = {c: 'w_{}'.format(c) for c in unweighted_linkpred_cols}
    weighted_linkpred_scores_df = weighted_linkpred_scores_df.rename(columns=cols_map)
    weighted_linkpred_cols = list(weighted_linkpred_scores_df.columns)

    hyg_scores_df = get_hypergraph_scores(weighted_lp_data, hypergraph_score_indices, include_train=include_train)
    hyg_scores_cols = list(hyg_scores_df.columns)
    scores_df = pd.merge(unweighted_linkpred_scores_df, weighted_linkpred_scores_df, left_index=True, right_index=True)
    if not hyg_scores_df.empty:
        scores_df = pd.merge(scores_df, hyg_scores_df, left_index=True, right_index=True)
    test_pos_pairs = set(zip(*weighted_lp_data['A_test_pos'].nonzero()))
    pos_pairs = test_pos_pairs if not include_train else test_pos_pairs.union(
        set(zip(*weighted_lp_data['A_train'].nonzero())))
    scores_df['label'] = scores_df.index.map(lambda x: int(x in pos_pairs))
    try:
        perf_df = get_perf_df(scores_df, unweighted_linkpred_cols + weighted_linkpred_cols, hyg_scores_cols)
    except ValueError:
        perf_df = None
    return data, weighted_lp_data, \
           {'scores': scores_df, 'perf': perf_df}


def populate_and_store_tables(data_names, split_modes, predictor_cols, metrics, path):
    tables = {}
    for split_mode in split_modes:
        tables[split_mode] = defaultdict(list)
        data_params['split_mode'] = split_mode
        for data_name in data_names:
            data_params['data_name'] = data_name
            _, _, lp_results = perform_link_prediction(data_params, lp_data_params, lp_params)
            for metric in metrics:
                row = lp_results['perf'].loc[metric, predictor_cols]
                row.name = get_data_abbr(data_name)
                tables[split_mode][metric].append(row)

    for split_mode in split_modes:
        for metric in metrics:
            df = pd.DataFrame(tables[split_mode][metric]).T
            df = df.rename_axis(metric, axis=1)
            df = df.rename(columns={c: '\\textbf{{{}}}'.format(c) for c in df.columns})
            tables[split_mode][metric] = df
            file_path = os.path.join(base_path, 'tables', path, '_'.join([split_mode, metric]) + '.tex')
            df.to_latex(open(file_path, 'w'), bold_rows=True, escape=False)
    #             print(df.to_latex(bold_rows=True, escape=False))
    return tables


def find_rank_correlation_matrix(data_params, lp_data_params, lp_params):
    perf_df, a, b = perform_link_prediction(data_params, lp_data_params, lp_params)
    df = b['scores']
    colss = df.columns
    mat = np.zeros((len(colss), len(colss)))
    ii = 0
    for i in colss:
        jj = 0
        for j in colss:
            df1 = df[[i]]
            df2 = df[[j]]
            coef, p = kendalltau(df1, df2)
            mat[ii][jj] = coef
            jj = jj + 1
        ii = ii + 1
    return mat


def find_feat_imp(data_names, split_modes, feature_combinations_map, metrics, classifier, path):
    classifier_params['classifier'] = classifier
    tables = {}
    for split_mode in split_modes:
        tables[split_mode] = {comb_name: defaultdict(list) for comb_name in feature_combinations_map}
        data_params['split_mode'] = split_mode
        for data_name in data_names:
            data_params['data_name'] = data_name
            for comb_name, comb in feature_combinations_map.items():
                # tables[split_mode][comb_name] = defaultdict(list)
                classifier_params['features'] = comb
                perf_df, feat_imp, _ = perform_classification(data_params, lp_data_params, lp_params, classifier_params,
                                                              random_state=42)
                print(feat_imp.sort_values(["importance"], axis=0, ascending=False))


def main1():
    lp_cols = default_lp_cols
    hyper_cols = default_hyper_cols
    lp_tables = populate_and_store_tables(data_names, split_modes, lp_cols, metrics, 'perf/graph')
    hyg_tables = populate_and_store_tables(data_names, split_modes, hyper_cols, metrics, 'perf/hypergraph')

    for combination_name, cols in mixed_combinations_map.items():
        path_name = 'perf/combined/{}'.format(combination_name)
        mkdir_p(os.path.join(base_path, 'tables', path_name))
        combined_tables[combination_name] = populate_and_store_tables(data_names, split_modes, cols, metrics, path_name)

    mcm = mixed_combinations_map
    iter_var = 5
    random_list = [65, 26, 17, 98, 21]
    for j in range(iter_var):
        random_state = random_list[j]
        for i in lp_cols[:]:
            a = populate_and_store_classifier_tables(data_names, split_modes, {(i, 'G'): [i],
                                                                               (i, 'H'): mcm[i][1:],
                                                                               (i, 'G+H'): mcm[i]},
                                                     metrics, 'xgboost', path='perf/classifier/{}'.format(i),
                                                     random_state=random_state, iter_var=j)
        a = populate_and_store_classifier_tables(data_names, split_modes, {('full', 'G'): lp_cols,
                                                                           ('full', 'H'): hyper_cols,
                                                                           ('full', 'G+H'): lp_cols + hyper_cols},
                                                 metrics, 'xgboost', path='perf/classifier/{}'.format('full'),
                                                 random_state=30, iter_var=j)


def get_default_params():
    default_params = {'data_params': {'data_name': 'email-Enron',
                                      'base_path': base_path,
                                      'split_mode': 'temporal',
                                      'max_size_limit': 10},
                      'lp_data_params': {'rho': 0.2,
                                         'neg_factor': 5,
                                         'neg_mode': 'random'},
                      'lp_params': {'linkpred_indices': default_lp_indices,
                                    'hypergraph_score_indices': default_hyper_indices},
                      'iter_var': 0
                      }
    return default_params


def parse_params():
    default_params = get_default_params()
    parser = argparse.ArgumentParser(description='Link Predictor')
    parser.add_argument('-d', '--data_name', type=str, default=defaults['data_name'],
                        help='String for the data name; default={}'.format(defaults['data_name']))
    parser.add_argument('-b', '--base_path', type=str, default=defaults['base_path'],
                        help='Path where data is stored and where cache is to be stored; default={}'.format(
                            defaults['base_path']))
    parser.add_argument('-s', '--split_mode', type=str, default=defaults['split_mode'],
                        help='structural or temporal; default={}'.format(defaults['split_mode']))
    parser.add_argument('-i', '--iter_var', type=int, default=0,
                        help='A mandatory placeholder for multiple runs')
    args = parser.parse_args()
    params = default_params.copy()
    params['data_params']['data_name'] = args.data_name
    params['data_params']['base_path'] = args.base_path
    params['data_params']['split_mode'] = args.split_mode
    params['iter_var'] = args.iter_var
    return params


def perform_GWH_classification(params, G_feats, W_feats, H_feats, classifier):
    feat_combs = [G_feats, W_feats, H_feats, G_feats + H_feats, W_feats + H_feats]
    params['classifier_params'] = {'classifier': classifier}
    # print('Iterating over feature combinations...')
    classifier_outputs = {}
    for comb in (feat_combs):
        params['classifier_params']['features'] = comb
        classifier_outputs[tuple(comb)] = perform_classification(params['data_params'],
                                                                 params['lp_data_params'],
                                                                 params['lp_params'],
                                                                 params['classifier_params'],
                                                                 params['iter_var'])
    return classifier_outputs


def relocate_data(data):
    S, times, labels = data
    nodes = list(range(S.shape[0]))
    hyperedges = incidence_to_hyperedges(S)
    rel_hyperedges = []
    print('Relocating...')
    for f in tqdm(hyperedges):
        rel_f = sample(nodes, len(f))
        rel_hyperedges.append(frozenset(rel_f))
    print('Done!')
    rel_S = hyperedges_to_incidence(rel_hyperedges, S.shape[0])
    relocated_data = rel_S, times, labels
    return relocated_data


def perform_one_relocation(params, data, parallel_version):
    relocated_data = relocate_data(data)
    _, _, lp_results = perform_link_prediction(params['data_params'],
                                               params['lp_data_params'],
                                               params['lp_params'],
                                               params['iter_var'],
                                               include_train=True,
                                               data=relocated_data,
                                               parallel_version=parallel_version)
    return lp_results['perf']


def compare_rel_hyg_scores(params, num_relocations=2, parallel_version=False):
    data, _, lp_results = perform_link_prediction(params['data_params'],
                                                  params['lp_data_params'],
                                                  params['lp_params'],
                                                  params['iter_var'], include_train=True,
                                                  data=None,
                                                  parallel_version=parallel_version)
    perf = lp_results['perf']
    perfs_rel = []
    if not parallel_version:
        for i in tqdm(range(num_relocations), 'Relocation: '):
            print('Relocating {} out of {} times.'.format(i, num_relocations))
            perf_rel = perform_one_relocation(params, data, parallel_version)
            perfs_rel.append(perf_rel)
    else:
        max_workers = min(num_relocations, multiprocessing.cpu_count())
        pool = ProcessPoolExecutor(max_workers=max_workers)
        process_list = []
        for i in range(num_relocations):
            process_list.append(pool.submit(perform_one_relocation, params, data, parallel_version))
            print('{} of {} relocation processes scheduled'.format(len(process_list), num_relocations))
        for p in as_completed(process_list):
            perf_rel = p.result()
            perfs_rel.append(perf_rel)
            print('{} of {} relocation processes completed'.format(len(perfs_rel), len(process_list)))
        pool.shutdown(wait=True)
    return perf, perfs_rel


def main(mode='plp'):
    mode = 'crhs'
    params = parse_params()
    pprint(params)
    if mode == 'plp':
        _, _, lp_results = perform_link_prediction(params['data_params'],
                                                params['lp_data_params'],
                                                params['lp_params'],
                                                params['iter_var'])
    elif mode == 'pc':
        mcm = mixed_combinations_map
        # For individual 10 lp algorithms scores:
        print('For each LP col, finding GWH scores...')
        outputs = {}
        for lp_col in tqdm(default_lp_cols):
            G_feats = [lp_col]
            W_feats = ['w_{}'.format(lp_col)]
            H_feats = mcm[lp_col][1:]
            outputs[lp_col] = perform_GWH_classification(params, G_feats, W_feats, H_feats, 'xgboost')
        # For all scores at once:
        print('Finding GWH scores for all scores at once...')
        G_feats = default_lp_cols
        W_feats = ['w_{}'.format(c) for c in default_lp_cols]
        H_feats = default_hyper_cols
        perform_GWH_classification(params, G_feats, W_feats, H_feats, 'xgboost')

    elif mode == 'crhs':
        # Compare relocated hypergraph scores
        params['lp_data_params']['rho'] = 0.0
        params['lp_data_params']['neg_factor'] = -1
        params['lp_params']['hypergraph_score_indices'] = []
        params['lp_params']['graph_score_indices'] = list(range(len(all_predictor_names)))
        perf, perfs_rel = compare_rel_hyg_scores(params)
        print(perf, perfs_rel)


if __name__ == '__main__':
    main('pc')
