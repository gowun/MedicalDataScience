import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from metrics import *

def logiReg_model_with_performance(train_X_y, max_iter, valid_X_y=None, class_weight=None):
    X, y = train_X_y
    clf = LogisticRegression(class_weight=class_weight, max_iter=max_iter, random_state=1234).fit(X, y)
    if valid_X_y is None:
        vX = X
        vy = y
    else:
        vX, vy = valid_X_y
    pred = clf.predict(vX)
    prob = clf.predict_proba(vX)[:, 1]
    
    performance = compute_performance(vy, pred, prob)

    result = dict()
    result['model'] = clf
    result['coefficients'] = pd.DataFrame({'feature': [''] + list(X.columns), 'coefficient': list(clf.intercept_) + list(clf.coef_[0])})
    result['performance'] = performance
    result['columns'] = X.columns

    return result
