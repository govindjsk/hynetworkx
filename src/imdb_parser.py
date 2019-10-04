from joblib import Memory
import os
import pandas as pd
from tqdm import tqdm_notebook
from scipy.sparse import csr_matrix, triu
import random

data_path = '/content/gdrive/My Drive/Colab Notebooks/data/imdb'
file_names = {'names': 'name_basics.tsv', 'titles': 'title_basics.tsv', 'name_title': 'title_principals.tsv',
              'title_akas': 'title_akas.tsv'}
cachedir = os.path.join(data_path, 'cache')
memory = Memory(cachedir, verbose=0)


@memory.cache
def read_actors():
    df = pd.read_csv(os.path.join(data_path, file_names['names']), sep='\t')[['nconst', 'primaryName']]
    actor_name_map = dict(zip(df['nconst'], df['primaryName']))
    return actor_name_map


@memory.cache
def read_movies():
    df = pd.read_csv(os.path.join(data_path, file_names['titles']), sep='\t')
    nonadult_movie_filter = (df['titleType'] == 'movie') & (df['isAdult'] == 0)
    df = df[nonadult_movie_filter].drop(['titleType', 'isAdult'], axis=1)
    movie_cols = ['tconst', 'originalTitle', 'startYear']
    df = df[movie_cols].rename(columns={'tconst': 'mid', 'originalTitle': 'title', 'startYear': 'year'})
    df['year'] = df['year'].apply(lambda x: 0 if x == '\\N' else x)
    movie_name_map = dict(zip(df['mid'], df['title']))
    movie_year_map = dict(zip(df['mid'], df['year']))
    return movie_name_map, movie_year_map


@memory.cache
def get_movie_actor_df():
    df = pd.read_csv(os.path.join(data_path, file_names['name_title']), sep='\t')
    actor_filter = (df['category'].isin(['actor', 'actress']))
    df = df[actor_filter]
    cols = ['tconst', 'nconst']
    df = df[cols].rename(columns={'tconst': 'mid', 'nconst': 'aid'})
    return df


@memory.cache
def get_lang_movie_ids(languages=None):
    if languages is None:
        languages = ['en', 'hi']
    df = pd.read_csv(os.path.join(data_path, file_names['title_akas']), sep='\t')
    cols = ['titleId', 'region', 'language', 'isOriginalTitle']
    df = df[cols]
    df = df.rename(columns={'titleId': 'mid', 'language': 'lang', 'isOriginalTitle': 'is_orig'})
    lang_movie_ids_map = {lang: set(df[(df['lang'] == lang)]['mid'].unique()) for lang in languages}
    return lang_movie_ids_map


lang_movie_ids = get_lang_movie_ids()


def preprocess_data(an_map, mn_map, my_map, am_df, lang=None):
    aids = set(an_map.keys()).intersection(set(am_df['aid']))
    mids = set(mn_map.keys()).intersection(set(am_df['mid']))
    if lang:
        mids = lang_movie_ids[lang].intersection(mids)
    am_df = am_df[(am_df['aid'].isin(aids)) & (am_df['mid'].isin(mids))]

    aids = list(sorted(set(am_df['aid'])))
    mids = list(sorted(set(am_df['mid'])))

    old_new_aid_map = {aids[i]: i for i in range(len(aids))}
    old_new_mid_map = {mids[i]: i for i in range(len(mids))}

    an_map = {i: an_map[aids[i]] for i in range(len(aids))}
    mn_map = {i: mn_map[mids[i]] for i in range(len(mids))}
    my_map = {i: my_map[mids[i]] for i in range(len(mids))}

    am_df['aid'] = am_df['aid'].apply(lambda x: old_new_aid_map[x])
    am_df['mid'] = am_df['mid'].apply(lambda x: old_new_mid_map[x])
    return an_map, mn_map, my_map, am_df


def get_incidence_matrix(am_df, mn_map=None):
    movie_actors_df = am_df.groupby('mid').agg(lambda x: tuple(sorted(x))).reset_index().rename(columns={'aid': 'aids'})
    hyperedges = list(set(movie_actors_df['aids']))
    movie_actors_df['m_name'] = movie_actors_df['mid'].apply(lambda x: mn_map[x])
    df = movie_actors_df[['aids', 'm_name']].groupby('aids').agg(lambda x: tuple(sorted(x))).reset_index().rename(
        columns={'m_name': 'm_names'})
    hyperedge_labels = dict(zip(df['aids'], df['m_names']))
    hyperedge_labels = {k: '; '.join(v) for k, v in hyperedge_labels.items()}
    I = []
    J = []
    m = len(hyperedges)
    F_labels = {}
    for j, he in tqdm_notebook(enumerate(hyperedges)):
        label = hyperedge_labels[he]
        F_labels[j] = label
        he = list(he)
        I.extend(he)
        J.extend([j] * len(he))
    n = max(I) + 1
    S = csr_matrix(([1] * len(I), (I, J)), shape=(n, m))
    return S, F_labels


def sample_pos(A, rho):
    edges = set(zip(*triu(A).nonzero()))
    test_edges = random.sample(edges, int(rho * len(edges)))
    test_I, test_J = zip(*test_edges)
    A_pos = csr_matrix(([1] * len(test_I + test_J), (test_I + test_J, test_J + test_I)), shape=A.shape)
    return A_pos


@memory.cache
def get_preprocessed_data(lang=None):
    print('Reading actors')
    an_map = read_actors()
    print('Reading movies')
    mn_map, my_map = read_movies()
    print('Reading movie_actor data')
    am_df = get_movie_actor_df()
    print('Preprocessing data')
    an_map, mn_map, my_map, am_df = preprocess_data(an_map, mn_map, my_map, am_df, lang=lang)
    return an_map, mn_map, my_map, am_df
