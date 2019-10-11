from collections import defaultdict

import pandas as pd

from src.data_preparer import filter_size, prepare_lp_data
from src.hypergraph_link_predictor import get_hypergraph_scores
from src.link_predictor import get_perf_df
from src.linkpred_predictor import get_linkpred_scores
from src.supervised_link_predictor import classify
from src.utils import get_data_abbr, mkdir_p
from .incidence_matrix import parse_benson_incidence_matrix as parse_S

from joblib import Memory
import os

base_path = '/content/gdrive/My Drive/Colab Notebooks/data/'
cachedir = os.path.join(base_path, 'cache')
memory = Memory(cachedir, verbose=0)

data_params = {'data_name': 'contact-high-school',
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

lp_cols = ['AA', 'AS', 'CN', 'Cos', 'PA', 'JC', 'MxO', 'MnO', 'NM', 'Prn']
metrics = ['ap', 'auc', 'p@+', 'p@10', 'p@100', 'r@+', 'r@10', 'r@100']
hyper_cols = ['HAAM', 'HAAa', 'HAAl1', 'HAAl2',
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


@memory.cache
def perform_classification(data_params, lp_data_params, lp_params, classifier_params, random_state=42, iter_var=0):
    _, lp_results = perform_link_prediction(data_params, lp_data_params, lp_params, iter_var)
    features, classifier = [classifier_params[x] for x in ['features', 'classifier']]
    perf_df, feat_imp_df, classifier_scores_df = classify(lp_results, features, classifier, random_state, iter_var)
    return perf_df, feat_imp_df, classifier_scores_df


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


@memory.cache
def perform_link_prediction(data_params, lp_data_params, lp_params=None, iter_var=0):
    """
    data_params: {'data_name', 'base_path', 'split_mode', 'max_size_limit'}
    lp_data_params: {'rho', 'neg_factor', 'neg_mode', 'weighted_flag'}
    lp_params: {'linkpred_indices', 'hypergraph_score_indices'}

    returns: (data, lp_data, lp_results)
    """
    print('READING DATASET...')
    data_name, base_path, split_mode, max_size_limit = [data_params[x] for x in
                                                        ['data_name', 'base_path', 'split_mode', 'max_size_limit']]

    print('PREPARING LP DATA...')
    rho, neg_factor, neg_mode, weighted_flag = [lp_data_params[x] for x in
                                                ['rho', 'neg_factor', 'neg_mode', 'weighted_flag']]

    S, times, id_label_map = parse_S(data_name, base_path=base_path,
                                     ignore_time=(split_mode == 'structural'))

    if split_mode == 'structural':
        S, times = filter_size(S, times, 2, max_size_limit)

    lp_data = prepare_lp_data(S, weighted_flag, times, rho, neg_factor, neg_mode)

    print('PERFORMING LINK PREDICTION...')
    if lp_params:
        linkpred_indices, hypergraph_score_indices = [lp_params[x] for x in
                                                      ['linkpred_indices', 'hypergraph_score_indices']]
    else:
        linkpred_indices, hypergraph_score_indices = None, None
    linkpred_scores_df = get_linkpred_scores(lp_data, weighted_flag, linkpred_indices)
    linkpred_cols = list(linkpred_scores_df.columns)
    hyg_scores_df = get_hypergraph_scores(lp_data, hypergraph_score_indices)
    hyg_scores_cols = list(hyg_scores_df.columns)
    scores_df = pd.merge(linkpred_scores_df, hyg_scores_df, left_index=True, right_index=True)
    pos_pairs = set(zip(*lp_data['A_test_pos'].nonzero()))
    scores_df['label'] = scores_df.index.map(lambda x: int(x in pos_pairs))
    perf_df = get_perf_df(scores_df, linkpred_cols, hyg_scores_cols)
    return lp_data, \
           {'scores': scores_df, 'perf': perf_df}


def populate_and_store_tables(data_names, split_modes, predictor_cols, metrics, path):
    tables = {}
    for split_mode in split_modes:
        tables[split_mode] = defaultdict(list)
        data_params['split_mode'] = split_mode
        for data_name in data_names:
            data_params['data_name'] = data_name
            _, lp_results = perform_link_prediction(data_params, lp_data_params, lp_params)
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


def main():
    lp_tables = populate_and_store_tables(data_names, split_modes, lp_cols, metrics, 'perf/graph')
    hyg_tables = populate_and_store_tables(data_names, split_modes, hyper_cols, metrics, 'perf/hypergraph')

    for combination_name, cols in mixed_combinations_map.items():
        path_name = 'perf/combined/{}'.format(combination_name)
        mkdir_p(os.path.join(base_path, 'tables', path_name))
        combined_tables[combination_name] = populate_and_store_tables(data_names, split_modes, cols, metrics, path_name)


if __name__ == '__main__':
    main()
