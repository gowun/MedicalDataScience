import pandas as pd
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import linkage
from .utils import *


sns.set(color_codes=True)


def max_len(letter, split_str):
  tfs = np.array(split_str) == letter
  len_list = []
  ln = 0
  for i in tfs:
    if i:
      ln += 1
    else:
      len_list.append(ln)
      ln = 0
  len_list.append(ln)
  return max(len_list)


def find_max_len_by_ch(string, ch_list, splt='_'):
  split_str = string.split(splt)
  return list(map(lambda x: max_len(x, split_str), ch_list))


def compute_linkage_impurity(link_info, class_labels):
  def find_values(LRmode, which, labels):
    v = int(link_info.iloc[which][LRmode])
    if v in range(len(class_labels)):
      labels[LRmode][which] = class_labels[v]
    else:
      which_ = list(link_info['new_level']).index(v)
      labels[LRmode][which] = '_'.join([labels['left'][which_], labels['right'][which_]])
    return labels
  
  link_info['new_level'] = list(range(len(class_labels), len(class_labels) + len(link_info)))
  labels = dict()
  labels['left'] = [''] * len(link_info)
  labels['right'] = [''] * len(link_info)

  done = np.array([False] * len(link_info))
  level = 2
  while sum(done) < len(link_info):
    idx = link_info.loc[link_info['level'] == level].index
    for i in idx:
      labels = find_values('left', i, labels)
      labels = find_values('right', i, labels)
      done[i] = True
    level += 1
  link_info['left_label'] = labels['left']
  link_info['right_label'] = labels['right']

  lb = sorted(set(class_labels))
  cnt_lb = dict()
  targets = []
  for l in lb:
    cnt_lb[l] = sum(np.array(class_labels) == l)
    targets.append('_'.join([l] * cnt_lb[l]))
  
  str_arr = link_info[['left_label', 'right_label']].values 
  for ss in str_arr:
    in_tfs = list(map(lambda s: list(map(lambda x: x in s, targets)), ss))
    in_tfs = in_tfs[0] + in_tfs[1]
    if True in in_tfs:
      break
  try:
    which = in_tfs.index(True)
    if which < len(targets):
      tt = targets[which]
      s = ss[0]
    else:
      tt = targets[which - len(targets)]
      s = ss[1]
    if tt == s:
      impurity = 0
    else:
      l1 = len(s.split('_'))
      l2 = len(tt.split('_'))
      impurity = l1 - l2
  except ValueError:
    impurity = len(class_labels)
  
  return impurity, link_info


def compute_cluster_impurity(df, class_labels):
    links = pd.DataFrame(linkage(df.values, method='average'), columns=['left', 'right', 'distance', 'level'])
    return compute_linkage_impurity(links, class_labels)


def find_features_of_lowest_impurity(feature_sets, df, class_labels):
  impurity_list = []
  for fs in feature_sets:
    im, _ = compute_cluster_impurity(df[fs], class_labels)
    impurity_list.append(im)

  result = dict()
  result['impurity'] = min(impurity_list)
  result['features'] = list(feature_sets)[impurity_list.index(result['impurity'])]
  return result


def plot_cluster_heatmap(df, transpose=False):
    if transpose:
        df = transpose_df(df)
    df = df.astype(float)
    sns.clustermap(df)


def plot_sequential_cluster_heatmap(df, var_list, transpose=False):
    for vs in var_list:
        ins = list(map(lambda x: x in df.columns, vs))
        var = np.array(vs)[ins]
        plot_cluster_heatmap(df[var], transpose=transpose)