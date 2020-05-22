import multiprocessing
import os
import sys
from concurrent.futures import as_completed
from concurrent.futures.process import ProcessPoolExecutor

from utils import get_library_path

library_path = get_library_path()
sys.path.append(library_path)
sys.path.append(os.path.join(library_path, 'hynetworkx'))
from linkpred import linkpred
import pandas as pd
from scipy.sparse import triu
import networkx as nx
import pickle
from tqdm.autonotebook import tqdm

base_path = '/content/gdrive/My Drive/Colab Notebooks/data/'

all_predictor_names = ['AdamicAdar',  # 0
                       'AssociationStrength',  # 1
                       'CommonNeighbours',  # 2
                       'Community',  # 3
                       'Copy',  # 4
                       'Cosine',  # 5
                       'DegreeProduct',  # 6
                       'GraphDistance',  # 7
                       'Jaccard',  # 8
                       'Katz',  # 9
                       'MaxOverlap',  # 10
                       'MinOverlap',  # 11
                       'NMeasure',  # 12
                       'Pearson',  # 13
                       'Random',  # 14
                       'ResourceAllocation',  # 15
                       'RootedPageRank',  # 16
                       'SimRank',  # 17
                       ]

# all_predictor_names = ['Random','AdamicAdar', 'CommonNeighbours', 'Cosine', 'DegreeProduct', 'Jaccard', 'Katz',
# 'SimRank']
predictor_abbr_map = {'AdamicAdar': 'AA',
                      'AssociationStrength': 'AS',
                      'CommonNeighbours': 'CN',
                      'Community': 'Comm',
                      'Copy': 'Cpy',
                      'Cosine': 'Cos',
                      'DegreeProduct': 'PA',
                      'GraphDistance': 'GD',
                      'Jaccard': 'JC',
                      'Katz': 'Kz',
                      'MaxOverlap': 'MxO',
                      'MinOverlap': 'MnO',
                      'NMeasure': 'NM',
                      'Pearson': 'Prn',
                      'Random': 'Rnd',
                      'ResourceAllocation': 'RA',
                      'RootedPageRank': 'RPR',
                      'SimRank': 'SR'}

all_predictors = [eval('linkpred.predictors.{}'.format(x)) for x in all_predictor_names]


def perform_one_lp(predictor_name, predictor, G_train, pairs, weighted):
    abbr = predictor_abbr_map[predictor_name]
    pred = predictor(G_train, strictly_included=pairs)
    try:
        results = pred.predict(weight='weight') if weighted else pred.predict()
    except TypeError:
        print("predict() got an unexpected keyword argument 'weight'")
        results = pred.predict()
    scores = {k: results[k] for k in pairs}
    return abbr, scores


def get_linkpred_scores(lp_data, weighted, predictor_indices=None, include_train=False, parallel_version=False):
    predictor_indices = predictor_indices or range(len(all_predictors))
    predictors = [all_predictors[i] for i in predictor_indices]
    predictor_names = [all_predictor_names[i] for i in predictor_indices]
    A_train = lp_data['A_train']
    A_test = lp_data['A_test']
    A_test_pos = lp_data['A_test_pos']
    A_test_neg = lp_data['A_test_neg']
    G_train = nx.from_scipy_sparse_matrix(A_train)
    if A_test_neg is None:
        pairs = list(itertools.combinations(range(A_train.shape[0]), 2))
    else:
        test_pairs = list(zip(*triu(A_test_pos + A_test_neg).nonzero()))
        pairs = test_pairs if not include_train else test_pairs + list(zip(*triu(A_train).nonzero()))
    scores = {}
    if not parallel_version:
        for i, predictor in tqdm(enumerate(predictors), 'Predictor: '):
            predictor_name = predictor_names[i]
            abbr, lp_scores = perform_one_lp(predictor_name, predictor, G_train, pairs, weighted)
            scores[abbr] = lp_scores
    else:
        num_predictors = len(predictors)
        max_workers = min(num_predictors, multiprocessing.cpu_count())
        pool = ProcessPoolExecutor(max_workers=max_workers)
        process_list = []
        for i, predictor in enumerate(predictors):
            predictor_name = predictor_names[i]
            process_list.append(pool.submit(perform_one_lp, predictor_name, predictor, G_train, pairs, weighted))
            print('{} of {} processes scheduled ({})'.format(len(process_list), num_predictors, predictor_name))
        for p in as_completed(process_list):
            abbr, lp_scores = p.result()
            scores[abbr] = lp_scores
            print('{} of {} processes completed'.format(len(scores), len(process_list)))
        pool.shutdown(wait=True)
    scores_df = pd.DataFrame(scores)
    return scores_df


def store_linkpred_scores(G_train, test_pairs, file_prefix, predictor_indices=None):
    predictor_indices = predictor_indices or range(len(all_predictors))
    predictors = [all_predictors[i] for i in predictor_indices]
    predictor_names = [all_predictor_names[i] for i in predictor_indices]
    for i in tqdm(range(len(predictors)), 'Predictor: '):
        predictor = predictors[i]
        abbr = predictor_abbr_map[predictor_names[i]]
        # print('Preparing predictor {}'.format(abbr))
        pred = predictor(G_train, strictly_included=test_pairs)
        # print('Performing prediction...')
        scores = pred.predict()
        scores = [scores[k] for k in test_pairs]
        # print('Done')
        file_path = file_prefix + '_' + abbr
        # print('Pickling into {}'.format(file_path))
        pickle.dump(scores, open(file_path, 'wb'))
        del scores
