import numpy as np 
import pandas as pd 
import os
from itertools import combinations
from itertools import chain
from ..modeling_tools import preprocessing as pp 
from ..modeling_tools import tree_modeling as tm 
from ..modeling_tools import linear_modeling as lm 
from ..modeling_tools import metrics as mt 
from ..modeling_tools import survival_analysis as sa 
from ..modeling_tools import utils as ut 
from ..modeling_tools import figure_eda as fe 
from ..modeling_tools import clustering as cl 

class BladderCancerQuantSeq():
    def __init__(self):
        self.home_path = os.path.abspath('MedicalDataScience/BladderCancer_Shin/') + '/'
        self.org_df, self.nor_df = ut.load_data(self.home_path + 'two_data_v1.1.pkl', 'pickle')
        self.dists = ut.load_data(self.home_path + 'distributions.pkl', 'pickle')
        self.random_forest = ut.load_data(self.home_path + 'random_forest_model.pkl', 'pickle')
        self.y = list(map(lambda x: x.split('_')[-1], self.org_df.index))
        self.y_01 = np.array([0] * len(self.y))
        self.y_01[np.array(self.y) == 'R'] = 1
        tmp = ut.load_data(self.home_path + 'filtered_classifiers.csv', 'csv')
        self.classifiers = {}
        for c in tmp.columns:
            vv = tmp[c].values[[tmp[c].isna() == False]]
            self.classifiers[c] = vv
        self.models = ut.load_data(self.home_path + 'models_v1.1.pkl', 'pickle')
        self.score_df = ut.load_data(self.home_path + 'filtered_scores.csv', 'csv')
        self.performance_df = ut.load_data(self.home_path + 'filtered_compared_performance.csv', 'csv')

    
    def do_yourself_find_best_normalization(self, update=False):
        dists = pp.do_all_scalers(self.org_df)
        nor_df = pp.find_best_normalization(self.org_df, dists, self.y)
        if update:
            self.dists = dists
            self.nor_df = nor_df
        else:
            return self.nor_df


    def do_yourself_select_features_by_MI_VIF(self, upper_limit=1000):
        features_list = list(map(lambda x: self.random_forest['feature importance']['feature'].values[:x], range(5, 30)))
        classifiers = {'gowun': cl.find_features_of_lowest_impurity(features_list, self.nor_df, self.y)}
        tmp = ut.load_data(self.home_path + 'Classifiers.csv', 'csv')
        for c in tmp.columns[:-1]:
            vv = tmp[c].values[[tmp[c].isna() == False]]
            in_tf = list(map(lambda x: x in self.nor_df.columns, vv))
            vv = np.array(vv)[in_tf]
            classifiers[c] = vv
        filtered = {}
        for k, vs in classifiers.items():
            print('!!!!!' + k + '!!!!!')
            tmp = pp.filter_by_VIF_MI(self.nor_df, vs, self.y, upper_limit)
            print(tmp)
            filtered[k] = tmp['feature'].values
        return filtered 


    def do_yourself_best_feature_combination(self, filtered, mMin=3, update=False):
        best_comb = cl.find_best_feature_comb_parallel(self.nor_df, self.y, filtered, nMin=mMin)
        classifiers = {}
        classifier_df = pd.DataFrame()
        order = list(best_comb.keys())
        max_len = max(list(map(lambda x: len(x[1]), best_comb.items())))
        for k in order:
            classifier_df[k] = list(best_comb[k]) + [''] * (max_len - len(best_comb[k]))
            classifiers[k] = best_comb[k]
        if update:
            self.classifiers = classifiers
        else:
            return classifiers, classifier_df

    
    def do_yourself_modeling(self, update=False):
        models = {}
        for k, v in self.classifiers.items():
            md = lm.logiReg_model_with_performance([self.nor_df[v], self.y_01], 10, class_weight='balanced')
            md['scores'] = md['model'].predict_proba(self.nor_df[v])[:, 1]
            models[k] = md
        if update:
            self.models = models
        else:
            return models


    def do_yourself_compare_models(self, update=False):
        order = list(self.classifiers.keys())
        scores_df = pd.DataFrame(np.array(list(map(lambda x: x[1]['scores'], self.models.items()))).T, columns=order, index=self.org_df.index).reset_index()
        per_tb = pd.DataFrame(list(map(lambda x: x[1]['performance'], self.models.items())), index=order).reset_index()
        if update:
            self.score_df = scores_df
            self.performance_df = per_tb
        else:
            return scores_df, per_tb


    def plot_all_cluster_heatmap(self):
        self.nor_df.index = self.y 
        order = list(self.classifiers.keys())
        cl.plot_sequential_cluster_heatmap(self.nor_df, list(map(lambda x: x[1]['columns'], self.models.items())), True, order)