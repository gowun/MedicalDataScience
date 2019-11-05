import pandas as pd 
import numpy as np 
import os
import pickle

def save_as_file_colab(obj, path, mode):
  if mode == 'csv':
    nm = 'tmp.csv'
    obj.to_csv(nm, index=False)
  elif mode == 'pickle':
    nm = 'tmp.pkl'
    pickle.dump(obj, open(nm, 'wb'))
  os.system('cp ' + nm + " '"+ path + "'")


def load_data(path, mode):
    if mode == 'csv':
        return pd.read_csv(path)
    elif mode == 'pickle':
        return pickle.load(open(path, 'rb'))