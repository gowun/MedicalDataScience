import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import SMOTE 
import shap
import graphviz


def preprocessing_numeric(df_numeric, odd_values, rep_value=-999):
    def inconvertable(x):
        try:
            int(x)
            return False
        except:
            return True
    
    if rep_value == 'neighbor_mean':
        nn = NearestNeighbors().fit(df_numeric).kneighbors(df_numeric, 5)[1][:, 1:]
    
    for c in df_numeric.columns:
        # 문자형 -> 숫자형
        if df_numeric[c].dtypes == 'O':
            try:
                df_numeric[c] = df_numeric[c].astype(int)
            except:
                tmp = list(map(lambda x: inconvertable(x), df_numeric[c]))
                df_numeric[c].loc[tmp] = str(rep_value)
                df_numeric[c] = df_numeric[c].astype(int)

        if rep_value == 'neighbor_mean':
            odd_idx = sorted(set(list(df_numeric[c].isna().index) + list((df_numeric[c] in odd_values).index)))
            df_numeric[c].iloc[odd_idx] = list(map(lambda x: df_numeric[c].iloc[x].mean(), nn[odd_idx]))
        else:
            df_numeric[c] = df_numeric[c].fillna(rep_value)
            df_numeric[c].loc[df_numeric[c] in odd_values] = rep_value

    return df_numeric


def preprocessing_category(df_cat):
    for c in df_cat.columns:
        if 'category' in c:
            df_cat[c] = df_cat[c].fillna(0)
        else:
            df_cat[c] = df_cat[c].fillna(999)

    encoder = OneHotEncoder().fit(df_cat)
    onehot = pd.DataFrame(encoder.transform(df_cat).toarray(), columns=encoder.get_feature_names())
    rm_cols = list(filter(lambda x: x.endswith('999.0'), onehot.columns))
    onehot = onehot.drop(rm_cols, 1)
    new_cols = []
    for i, c in enumerate(df_cat.columns):
        mini = list(filter(lambda x: str(i) in x.split('_')[0], onehot.columns))
        for cc in mini:
            cc = cc.split('_')
            cc[0] = c
            new_cols += ['_'.join(cc)]
    onehot.columns = new_cols
    
    return onehot


def random_oversampling(idx_list, n):
    np.random.seed(1234)
    return np.random.choice(idx_list, n)


def SMOTE_oversampling(X, y, k_neighbors, n_samples, target_ratios=[0.5, 0.5]):
    n_original = pd.value_counts(y)[[0, 1]]
    n_samples_by_class = list()
    over_X = list()
    over_y = list()
    for i in range(2):
        n_samples_by_class.append(round(n_samples * target_ratios[i]))
        over_X.append(X.loc[y == i].reset_index(drop=True))
        over_y.append(np.array(y)[y == i])
    
    sm = SMOTE(random_state=1234, k_neighbors=k_neighbors, sampling_strategy='all')
    iteration = 0
    done = [False, False]
    while True:
        over_lens = list(pd.value_counts(y)[[0, 1]])
        majority = over_lens.index(max(over_lens))
        X_res, y_res = sm.fit_resample(X, y)
        X_res = pd.DataFrame(X_res, columns=X.columns)
        for i in range(2):
            tf = np.array(y_res == i)[len(X):]
            tmp = X_res.iloc[len(X):].loc[tf]
            over_X[i] = pd.concat([over_X[i], tmp]).reset_index(drop=True)
            over_y[i] = np.concatenate([over_y[i], y_res[len(X):][tf]])
            if len(over_X[i]) >= n_samples_by_class[i]:
                done[i] = True
        print(str(iteration) + ": oversampled on class " + str(1 - majority) + " of totaly " + str(sum(map(lambda x: len(x), over_y))) + ' = ' + str(list(map(lambda x: len(x), over_y))))
        print(done)

        if sum(done) == 2:
            over_X, over_y = np.array(list(map(lambda x: (x[0][:x[2]], x[1][:x[2]]), zip(over_X, over_y, n_samples_by_class)))).T
            over_X = pd.concat(over_X).reset_index(drop=True)
            over_y = np.concatenate(over_y)
            return over_X, over_y, n_original
        else:
            if iteration > 0:
                over_lens = list(map(lambda x: len(x), over_y))
                if done[majority] == True:
                    majority = 1 - majority
                    print("majority: " + str(majority))
                ratio = min(over_lens)/sum(over_lens)
                maj_idx = list(over_X[majority].index)
                np.random.seed(1234)
                under_idx = np.random.choice(maj_idx, int(len(maj_idx) * ratio))
                X = pd.concat([over_X[majority].iloc[under_idx], over_X[1 - majority]]).reset_index(drop=True)
                y = np.concatenate([over_y[majority][under_idx], over_y[1 - majority]])
                print(len(X), len(under_idx))
            iteration += 1


def binning(df, minmax_by_col_dict):
    for c in minmax_by_col_dict.keys():
        mnmx_lst = minmax_by_col_dict[c]
        for i, l in enumerate(mnmx_lst):
            mn, mx = l
            if mn == mx:
                tf = df[c] == mn
            else:
                tf = (df[c] > mn) * (df[c] <= mx)
            df[c].loc[tf] = i
    return df


def compute_shap_value(model, X):
    tree_explainer = shap.TreeExplainer(model)
    shap_values = tree_explainer.shap_values(X)[1]

    return shap_values


def each_decision_plot(ith, X, shap_values, y_ratio):
    shap.decision_plot(y_ratio, shap_values[ith], X.iloc[ith], link='logit')


def each_force_plot(ith, X, shap_values, y_ratio):
    shap.force_plot(y_ratio, shap_values[ith], X.iloc[ith], link='logit', matplotlib=True)