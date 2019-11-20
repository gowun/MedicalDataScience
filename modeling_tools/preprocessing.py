import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_selection import mutual_info_classif#, mutual_info_regression, SelectKBest
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from scipy.stats import normaltest
from statsmodels.stats.outliers_influence import variance_inflation_factor
from itertools import chain
from .figure_eda import *


def do_all_scalers(X):
  X = X.astype(float)
  X_log2 = pd.DataFrame()
  for c in X.columns:
      X_log2[c] = np.log2(X[c]).apply(lambda x: max([x, 0.0]))
  distributions = {
      'Unscaled data': X.values,
      'Data after log2 normalizing': X_log2.values,
      'Data after log2 norm. -> minmax_scaling': MinMaxScaler().fit_transform(X_log2),
      'Data after standard scaling': StandardScaler().fit_transform(X),
      'Data after min-max scaling': MinMaxScaler().fit_transform(X),
      'Data after max-abs scaling': MaxAbsScaler().fit_transform(X),
      'Data after robust scaling': RobustScaler(quantile_range=(25, 75)).fit_transform(X),
      'Data after power transformation (Yeo-Johnson)': PowerTransformer(method='yeo-johnson').fit_transform(X),
      'Data after quantile transformation (gaussian pdf)': QuantileTransformer(output_distribution='normal').fit_transform(X),
      'Data after sample-wise L2 normalizing': Normalizer().fit_transform(X)
      }
  
  return distributions


def check_varied_normality(distributions, ith_feature, feature_name, data_labels=None, draw=False):
  normal = dict()
  for i in distributions.keys():
    method = i
    arr1d = distributions[i][:, ith_feature]
    normal[method] = normaltest(arr1d)
    if draw:
      plot_histogram(arr1d, 'stack', data_labels, feature_name, method)

  return normal


def find_normality_degree(df, distributions, class_labels):
    method_scores = {}
    for i, col in enumerate(df.columns):
        tmp = check_varied_normality(distributions, i, col, class_labels)
        for k, v in tmp.items():
            if v[1] >= 0.05:
                if k in method_scores.keys():
                    method_scores[k] += 1
                else:
                    method_scores[k] = 1
    return method_scores


def find_best_normalization(df, distributions, class_labels):
    method_scores = find_normality_degree(df, distributions, class_labels)
    tmp = list(method_scores.values())
    max_idx = tmp.index(max(tmp))
    method = list(method_scores.keys())[max_idx]
    print(method_scores)
    nor_df = pd.DataFrame(distributions[method], columns=df.columns, index=df.index)
    return nor_df


def filter_by_VIF_MI(df, features, y, upper_limit=100, nMin=5):
    def part_calc(vs):
        total = pd.DataFrame({'feature': vs, 'MI': mutual_info_classif(df[vs], y, n_neighbors=len(vs), random_state=1234)})
        total['VIF'] = list(map(lambda x: variance_inflation_factor(df[vs].values, x), range(len(vs))))

        idx = list(total.loc[total['MI'] > 0.000001].loc[total['VIF'] <= upper_limit].index)
        if len(idx) < nMin:
            total['MI/VIF'] = total['MI'] / total['VIF']
            last = total.drop(idx).sort_values(by='MI/VIF', ascending=False)[:nMin-len(idx)].index 
            idx += list(last)
        return total.iloc[idx]

    np.random.seed(1234)
    in_tf = list(map(lambda x: x in df.columns, features))
    vvs = np.array(features)[in_tf]
    while len(vvs) > 16:  ## since VIF can process only 16 variables at the same time
        vvs = np.random.choice(vvs, len(vvs), replace=False)
        if len(vvs) % 16 == 1:
            start_idxes = list(range(0, len(vvs), 15))
            end_idxes = start_idxes[1:] + [len(vvs)]
        else:
            start_idxes = list(range(0, len(vvs), 16))
            end_idxes = start_idxes[1:] + [len(vvs)]
    filtered = []
    for i, j in zip(start_idxes, end_idxes):
        tmp = part_calc(vvs[i:j])
        filtered.append(tmp['feature'].values)
    vvs = list(chain(*filtered))
    
    print(vvs)
    total = part_calc(vvs)
    return total


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
                if rep_value == 'neighbor_mean':
                    df_numeric[c].loc[tmp] = list(map(lambda x: str(round(df_numeric[c].iloc[x].mean())), nn[tmp]))
                else:
                    df_numeric[c].loc[tmp] = str(rep_value)
                df_numeric[c] = df_numeric[c].astype(int)
        odd_idx = np.array(df_numeric.index)[list(map(lambda x: x in odd_values, df_numeric[c]))]
        if rep_value == 'neighbor_mean':
            odd_idx = sorted(set(list(odd_idx) + list(np.array(df_numeric.index)[df_numeric[c].isna()])))
            df_numeric[c].iloc[odd_idx] = list(map(lambda x: df_numeric[c].iloc[x].mean(), nn[odd_idx]))
        else:
            df_numeric[c] = df_numeric[c].fillna(rep_value)
            df_numeric[c].iloc[odd_idx] = rep_value

    return df_numeric


def preprocessing_category(df_cat, given_rm_cols=None, given_colnames=None):
    df_cat = df_cat.fillna(9999)
    encoder = OneHotEncoder().fit(df_cat)
    onehot = pd.DataFrame(encoder.transform(df_cat).toarray(), columns=encoder.get_feature_names())
    rm_cols = list(filter(lambda x: x.endswith('9999.0'), onehot.columns))
    if given_rm_cols is not None:
        rm_cols += given_rm_cols
    onehot = onehot.drop(rm_cols, 1)

    if given_colnames is None:
        new_cols = []
        for i, c in enumerate(df_cat.columns):
            mini = list(filter(lambda x: str(i) in x.split('_')[0], onehot.columns))
            for cc in mini:
                cc = cc.split('_')
                cc[0] = c
                new_cols += ['_'.join(cc)]
    else:
        new_cols = given_colnames
    onehot.columns = new_cols
    
    return onehot


def scale_btw_01(df, given_scaler=None):
    if given_scaler is None:
        sc = MinMaxScaler().fit(df)
    else:
        sc = given_scaler
    cols = np.array(df.columns)
    df_nor = pd.DataFrame(sc.transform(df), columns=cols)    
    for c in cols[df_nor.max(0) > 1]:
        df_nor[c].loc[df_nor[c] > 1] = 1
    for c in cols[df_nor.min(0) < 0]:
        df_nor[c].loc[df_nor[c] < 0] = 0

    return df_nor, sc


def random_oversampling(idx_list, n):
    np.random.seed(1234)
    return np.random.choice(idx_list, n)


def make_surrogate_data(X, n_samples, target_ratios=[0.5, 0.5]):
    sam_y = np.random.choice([0, 1], n_samples, replace=True, p=target_ratios)
    dd = {'a': X.values}
    sam_X = []
    for i, col in enumerate(X.columns):
        tmp = check_varied_normality(dd, i, col)
        if tmp['a'][1] < 0.05 or len(set(X[col])) <= 2:
            ratios = pd.value_counts(X[col]) / len(X)
            idx = sorted(ratios.index)
            ratios = list(ratios[idx])
            sam_X.append(np.random.choice(idx, n_samples, replace=True, p=ratios))
        else:
            sam_X.append(np.random.normal(np.mean(X[col]), np.std(X[col]), n_samples))
    sam_X = pd.DataFrame(np.array(sam_X).T, columns=X.columns)

    return sam_X, sam_y


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


