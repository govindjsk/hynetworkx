from sklearn.metrics import average_precision_score
import pandas as pd
from scipy.sparse import triu
import os
import networkx as nx
import pickle
from sklearn.metrics import roc_auc_score, precision_score, recall_score, fbeta_score
from tqdm import tqdm_notebook
import sys
sys.path.append("/content/gdrive/My Drive/Colab Notebooks/libraries/")
sys.path.append("/content/gdrive/My Drive/Colab Notebooks/libraries/hynetworkx")
base_path = '/content/gdrive/My Drive/Colab Notebooks/data/'

from src.hypergraph_link_predictor import store_hypergraph_scores
from src.linkpred_predictor import store_linkpred_scores


def get_perf_df(scores_df, linkpred_cols, hypergraph_cols):
    true = list(scores_df['label'])
    n_pos = scores_df['label'].value_counts()[1]
    perf_scores = {}
    for c in linkpred_cols + hypergraph_cols:
        predicted = list(scores_df[c])
        ap_score = average_precision_score(true, predicted)
        auc_score = roc_auc_score(true, predicted)
        prec_at_10, rec_at_10, f1_at_10 = find_prec_rec_fbeta_at_k(scores_df, c, k=10)
        prec_at_100, rec_at_100, f1_at_100 = find_prec_rec_fbeta_at_k(scores_df, c, k=100)
        prec_at_pos, rec_at_pos, f1_at_pos = find_prec_rec_fbeta_at_k(scores_df, c, k=n_pos)
        perf_scores[c] = {'ap': ap_score,
                          'auc': auc_score,
                          'p@10': prec_at_10,
                          'r@10': rec_at_10,
                          'p@100': prec_at_100,
                          'r@100': rec_at_100,
                          'p@+': prec_at_pos,
                          'r@+': rec_at_pos}
        perf_scores[c] = {k: round(v * 100, 2) for k, v in perf_scores[c].items()}
    perf_scores_df = pd.DataFrame(perf_scores)
    return perf_scores_df


def find_prec_rec_fbeta_at_k(scores_df, c, k=10, beta=0.5):
    if k > scores_df.shape[0]:
        return (0, 0, 0)
    sorted_df = scores_df[['label', c]].sort_values(c, ascending=False).reset_index().drop(['level_0', 'level_1'],
                                                                                           axis=1)
    sorted_df['pred_label'] = 0
    sorted_df.iloc[0:k, 2] = 1
    true_labels = list(sorted_df['label'])
    pred_labels = list(sorted_df['pred_label'])
    prec = precision_score(true_labels, pred_labels)
    rec = recall_score(true_labels, pred_labels)
    f_beta = fbeta_score(true_labels, pred_labels, beta)
    return (prec, rec, f_beta)


def store_scores_and_label(lp_data, data_name, split_mode,
                           linkpred_indices=None,
                           hypergraph_score_indices=None, base_path='.'):
    file_prefix = os.path.join(base_path, data_name + '_' + split_mode)

    G_train = nx.from_scipy_sparse_matrix(lp_data['A_train'])

    pos_pairs = set(zip(*triu(lp_data['A_test_pos']).nonzero()))
    neg_pairs = set(zip(*triu(lp_data['A_test_neg']).nonzero()))
    test_pairs = list(pos_pairs.union(neg_pairs))
    pair_label_tuples = [(pair, 1 if pair in pos_pairs else 0) for pair in tqdm_notebook(test_pairs)]

    pickle.dump(pair_label_tuples, open(file_prefix + 'pair_labels.pkl', 'wb'))
    store_linkpred_scores(G_train, test_pairs, file_prefix, linkpred_indices)
    store_hypergraph_scores(lp_data['S_train'], test_pairs, file_prefix, hypergraph_score_indices)


