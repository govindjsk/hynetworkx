import argparse
import pickle
from experimenter import *
from utils import *
import pandas as pd
import numpy as np
import os
from tqdm.autonotebook import tqdm
mcm = mixed_combinations_map
if 'HDPa' in mcm['PA']:
    mcm['PA'].remove('HDPa')


def parse_data_name():
    parser = argparse.ArgumentParser(description='Link Predictor')
    parser.add_argument('-d', '--data_name', type=str, default='swc',
                        help='String for the data name; default={}'.format('swc'))
    args = parser.parse_args()
    data_name = args.data_name
    return data_name


if __name__ == '__main__':
    home_path = os.path.join(get_library_path(), 'hynetworkx')
    data_names = [
                  'swc',
                  'email-Enron',
                  'contact-high-school',
                  'contact-primary-school',
                  'NDC-substances',
                  'tags-math-sx',
                  'threads-math-sx',
                  'coauth-DBLP'
                 ]

    split_mode = 'temporal'
    params = get_default_params()
    params['data_params']['split_mode'] = split_mode
    params['data_params']['base_path'] = get_base_path()
    dfs, train_dfs, test_dfs, overfit_dfs = [], [], [], []
    params['iter_var'] = 0
    lp_data_params = params['lp_data_params']
    lp_params = params['lp_params']
    iter_var = params['iter_var']

    params['lp_data_params']['rho'] = 0.0
    params['lp_data_params']['neg_factor'] = -1
    params['lp_params']['hypergraph_score_indices'] = []
    params['lp_params']['linkpred_indices'] = [9, 6, 0, 17, 16, 8, 2, 15]

    data_dfs = []
    data_name = parse_data_name()
    # for data_name in tqdm(data_names, 'Data: '):
    print('DATASET: {}'.format(data_name))
    pkl_file_name = os.path.join(home_path, 'comparative_analysis_{}.pkl'.format(data_name))
    try:
        (df1, df) = pickle.load(open(pkl_file_name, 'rb'))
        print('Found pickled')
    except FileNotFoundError:
        print('Not found pickled')
        params['data_params']['data_name'] = data_name
        perf, perfs_rel = compare_rel_hyg_scores(params, 5, True)
        df1 = perf.iloc[[1], range(10)].T.rename(columns = {'auc': data_name})
        df = pd.concat(perfs_rel)
        df = df.groupby(df.index).agg(lambda x: '${} \\pm {}$'.\
                                      format('%2.4f' % round(np.mean(x), 4),
                                             '%2.4f' % round(np.std(x), 4))).iloc[[1], range(10)].T
        df = df.rename(columns = {'auc': data_name})
        pickle.dump((df1, df), open(pkl_file_name, 'wb'))
        # data_dfs.append((df1, df))
