import pandas as pd
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from src.link_predictor import get_perf_df


def classify(lp_results, predictor_cols, classifier, random_state=42, iter_var=0):
    df = lp_results['scores'].copy(deep=True)

    if predictor_cols is None:
        predictor_cols = list(df.columns[:-1])

    X, y = df.loc[:, predictor_cols], df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=random_state)

    if classifier == 'xgboost':
        data_dmatrix = xgb.DMatrix(data=X_train, label=y_train)
        xg_reg = xgb.XGBClassifier()
        xg_reg.fit(X_train, y_train)
        preds = xg_reg.predict(X_test)
        pred_prob = xg_reg.predict_proba(X_test)
        probs = []
        for i in pred_prob:
            probs.append(i[1])
        feat_imp_df = pd.DataFrame({'importance': dict(zip(list(X.columns), xg_reg.feature_importances_))})
        cols = [classifier]
        df = df.loc[X_test.index, :]
        df[classifier] = probs
        perf_df = get_perf_df(df[[classifier, 'label']], cols, [])
        # print(perf_df)
        # rmse = np.sqrt(mean_squared_error(y, preds))
        return perf_df, feat_imp_df, df[[classifier]]
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
