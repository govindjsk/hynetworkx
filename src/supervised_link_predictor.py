from collections import defaultdict

import pandas as pd
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
# from src.experimenter import classifier_params, data_params, perform_classification, lp_data_params, lp_params, lp_cols, \
#     hyper_cols, metrics
from src.link_predictor import get_perf_df


def classify(lp_results, predictor_cols, classifier, iter_var=0):
    df = lp_results['scores'].copy(deep=True)

    if predictor_cols is None:
        predictor_cols = list(df.columns[:-1])

    X, y = df.loc[:, predictor_cols], df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=iter_var)

    if classifier == 'xgboost':
        # data_dmatrix = xgb.DMatrix(data=X_train, label=y_train)
        xg_reg = xgb.XGBClassifier()
        xg_reg.fit(X_train, y_train)

        # test_preds = xg_reg.predict(X_test)
        # train_preds = xg_reg.predict(X_train)

        test_probs = [i[1] for i in xg_reg.predict_proba(X_test)]
        train_probs = [i[1] for i in xg_reg.predict_proba(X_train)]

        feat_imp_df = pd.DataFrame({'importance': dict(zip(list(X.columns), xg_reg.feature_importances_))})

        train_col = '{}_{}'.format(classifier, 'train')
        test_col = '{}_{}'.format(classifier, 'test')

        test_df = df.loc[X_test.index, :]
        train_df = df.loc[X_train.index, :]

        train_df[train_col] = train_probs
        test_df[test_col] = test_probs

        train_perf_df = get_perf_df(train_df[[train_col, 'label']], [train_col], [])
        test_perf_df = get_perf_df(test_df[[test_col, 'label']], [test_col], [])
        return {'train_perf': train_perf_df,
                'test_perf': test_perf_df,
                'test_scores': test_df[[test_col]],
                'feat_imp': feat_imp_df}

    elif classifier == 'decision_tree':
        clf = DecisionTreeClassifier()
        clf = clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        # print(clf.predict_proba(X_test))
        importances = clf.feature_importances_
        feat_imp_df = pd.DataFrame({'importance': dict(zip(list(X.columns), clf.feature_importances_))})
        df = df.loc[X_test.index, :]
        df['decision_tree'] = y_pred
        print(precision_score(y_test, y_pred))
        print(recall_score(y_test, y_pred))

    elif classifier == 'naive_bayes':
        model = GaussianNB()
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        pred_prob = model.predict_proba(X_test)
        probs = []
        for i in pred_prob:
            probs.append(i[1])
        # feat_imp_df = pd.DataFrame({'importance': dict(zip(list(X.columns) , model.feature_importances_))})
        cols = [classifier]
        df = df.loc[X_test.index, :]
        df[classifier] = probs
        perf_df = get_perf_df(df[[classifier, 'label']], cols, [])
        # print(perf_df)
        # rmse = np.sqrt(mean_squared_error(y, preds))
        # return perf_df, feat_imp_df, df[[classifier]]




def main():
    random_state = 42
    data_names = ['email-Enron', 'contact-high-school']
    split_modes = ['structural', 'temporal']
    find_feat_imp(data_names, split_modes, {('full', 'G'): lp_cols,
                                            ('full', 'H'): hyper_cols,
                                            ('full', 'G+H'): lp_cols + hyper_cols},
                  metrics, 'xgboost', path='perf/classifier/{}'.format('full'))


if __name__ == '__main__':
    main()
