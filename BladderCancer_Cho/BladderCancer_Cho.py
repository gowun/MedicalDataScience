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

class BladderCander():
    def __init__(self):
        home_path = os.path.abspath('MedicalDataScience/BladderCancer_Cho/') + '/'
        self.Xy = ut.load_data(home_path + 'Xy_for_tree.pkl', 'pickle')
        self.event, self.durations = ut.load_data(home_path + 'event_durations.pkl', 'pickle')
        self.selected_Xs_with_importance = ut.load_data(home_path + 'selected_Xs.csv', 'csv')
        self.col_combinations = ut.load_data(home_path + 'col_combinations.pkl', 'pickle')
        self.only_marker_combinations = ut.load_data(home_path + 'only_marker_combinations.pkl', 'pickle')
        self.scaler = ut.load_data(home_path + 'scaler.pkl', 'pickle')
        self.random_forest = ut.load_data(home_path + 'random_forest.pkl', 'pickle')
        self.decision_tree = ut.load_data(home_path + 'decision_tree.pkl', 'pickle')
        self.decision_tree_only_markers = ut.load_data(home_path + 'decision_tree_only_markers.pkl', 'pickle')
        self.logistic_regression = ut.load_data(home_path + 'logistic_regression.pkl', 'pickle')


    def make_varied_cases(self, change=False):
        basal = ['CK5/6', 'CK14']
        luminal = ['CK20', 'GATA3', 'FOXA1']
        combs = []
        for b in basal:
            for l in luminal:
                combs.append([b, l])
            tmp = list(combinations(luminal, 2))
            for t in tmp:
                combs.append([b] + list(t))
        X_nec_cols = list(filter(lambda x: sum(map(lambda y: y in x, basal + luminal)) == 0, self.selected_Xs_with_importance['feature']))

        suffixes = set(self.selected_Xs_with_importance['feature']) - set(X_nec_cols)
        suffixes = sorted(set(list(map(lambda x: x[x.index(' '):], suffixes))))
        suf_list = list()
        col_list = list()
        for i in range(len(suffixes)):
            suf_list.append(list(map(lambda x: list(map(lambda y: y + suffixes[i], x)), combs)))
            col_list += list(map(lambda x: X_nec_cols + x, suf_list[i]))
        suf_list = list(chain(*suf_list))

        if change:
            self.col_combinations = col_list
            self.only_marker_combinations = suf_list
        return col_list, suf_list


    def do_yourself_random_forest(self, nOver=100000, nIter=1000, change=False):
        over_idx = pp.random_oversampling(self.Xy[0].index, nOver)
        sample_leaf = round(nOver / len(self.Xy[0]) * 3/2)
        over_X = self.Xy[0].iloc[over_idx]
        over_y = self.Xy[1].iloc[over_idx]
        result = tm.random_forest_with_performance([over_X, over_y], nIter, 3, sample_leaf)
        if change:
            self.random_forest = result
        return result


    def do_yourself_decision_tree(self, max_depths=[3, 4], min_sample_leaf=5, min_auc=0.75):
        dur = self.durations.values.T[0]
        models = tm.train_and_filter_models(self.Xy, self.col_combinations, max_depths, min_sample_leaf, min_auc, self.event, dur, 0.05, self.Xy[0])
        return tm.select_models(models, 2, print_score=True)


    def do_yourself_decision_tree_only_markers(self, max_depths=[3, 4], min_sample_leaf=5, min_AUC=0.75):
        dur = self.durations.values.T[0]
        models = tm.train_and_filter_models(self.Xy, self.only_marker_combinations, max_depths, min_sample_leaf, min_AUC, self.event, dur, 0.05, self.Xy[0])
        return tm.select_models(models, 2, print_score=True)


    def do_yourself_logistic_regression(self, max_iter=10, min_auc=0.75):
        dur = self.durations.values.T[0]
        X_nor, _ = pp.scale_btw_01(self.Xy[0])
        return lm.train_and_filter_models([X_nor, self.Xy[1]], self.col_combinations, max_iter, min_auc, self.event, dur, 0.05, self.Xy[0])


    def make_score(self, mode, X):
        if mode == 'logistic_regression':
            X = pp.scale_btw_01(X, self.scaler)
            model = self.logistic_regression
        elif mode == 'decition_tree':
            model = self.decision_tree
        elif mode == 'decition_tree_only_markers':
            model = self.decision_tree_only_markers
        elif mode == 'random_forest':
            model = self.random_forest
        print(X.head())
        cols = list(model['columns'])
        return model['model'].predict_proba(X[cols])[:, 1]


    def draw_multiple_KM_graphs(self, durations, group, event):
        km_list = list()
        for c in durations.columns:
            km_list.append(sa.do_KM_analysis(durations[c].values, group, event, ['Pred_NR', 'Pred_R'], c))
    

    def do_validation(self, mode, X, y, cutoff=0.5):
        score = self.make_score(mode, X)
        pred = np.array([0] * len(y))
        pred[score >= cutoff] = 1
        performance = mt.compute_performance(y, pred, score)
        mt.draw_auc(y, [score])
        self.draw_multiple_KM_graphs(self.durations, pred, self.event)
        
        return performance


    def compare_auces(self):
        names = ['logistic_regression', 'decision_tree', 'decision_tree_only_markers', 'random_forest']
        scores = list()
        for md in names:
            scores.append(self.make_score(md, self.Xy[0]))
        
        mt.draw_auc(self.Xy[1], scores, label_list=names)


