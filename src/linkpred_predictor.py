import sys

sys.path.append("/content/gdrive/My Drive/Colab Notebooks/libraries/")
sys.path.append("/content/gdrive/My Drive/Colab Notebooks/libraries/hynetworkx")
from linkpred import linkpred
import pandas as pd
from scipy.sparse import triu
import networkx as nx
import pickle
from tqdm import tqdm_notebook
import sys

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


def get_linkpred_scores(lp_data, weighted, predictor_indices=None):
    predictor_indices = predictor_indices or range(len(all_predictors))
    predictors = [all_predictors[i] for i in predictor_indices]
    predictor_names = [all_predictor_names[i] for i in predictor_indices]
    A_train = lp_data['A_train']
    A_test = lp_data['A_test']
    A_test_pos = lp_data['A_test_pos']
    A_test_neg = lp_data['A_test_neg']
    G_train = nx.from_scipy_sparse_matrix(A_train)
    test_pairs = list(zip(*triu(A_test_pos + A_test_neg).nonzero()))
    scores = {}
    if weighted:
        for i in tqdm_notebook(range(len(predictors)), 'Predictor: '):
            predictor = predictors[i]
            abbr = predictor_abbr_map[predictor_names[i]]
            print('Preparing predictor {}'.format(abbr))
            pred = predictor(G_train, strictly_included=test_pairs)
            print('Performing prediction...')
            try:
                results = pred.predict(weight='weight')
            except TypeError:
                print("predict() got an unexpected keyword argument 'weight'")
                results = pred.predict()
            print('Done')
            scores[abbr] = {k: results[k] for k in test_pairs}
        scores_df = pd.DataFrame(scores)
    else:
        for i in tqdm_notebook(range(len(predictors)), 'Predictor: '):
            predictor = predictors[i]
            abbr = predictor_abbr_map[predictor_names[i]]
            print('Preparing predictor {}'.format(abbr))
            pred = predictor(G_train, strictly_included=test_pairs)
            print('Performing prediction...')
            results = pred.predict()
            print('Done')
            scores[abbr] = {k: results[k] for k in test_pairs}
        scores_df = pd.DataFrame(scores)
    return scores_df


def store_linkpred_scores(G_train, test_pairs, file_prefix, predictor_indices=None):
    predictor_indices = predictor_indices or range(len(all_predictors))
    predictors = [all_predictors[i] for i in predictor_indices]
    predictor_names = [all_predictor_names[i] for i in predictor_indices]
    for i in tqdm_notebook(range(len(predictors)), 'Predictor: '):
        predictor = predictors[i]
        abbr = predictor_abbr_map[predictor_names[i]]
        print('Preparing predictor {}'.format(abbr))
        pred = predictor(G_train, strictly_included=test_pairs)
        print('Performing prediction...')
        scores = pred.predict()
        scores = [scores[k] for k in test_pairs]
        print('Done')
        file_path = file_prefix + '_' + abbr
        print('Pickling into {}'.format(file_path))
        pickle.dump(scores, open(file_path, 'wb'))
        del scores
