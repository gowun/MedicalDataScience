import pandas as pd
import numpy as np
import plotly.graph_objects as go


## mode in ['overlay', 'stack']
def plot_histogram(arr1d, mode, data_labels=None, xaxis=None, title=None):
  fig = go.Figure()
  if data_labels is not None:
    labels = sorted(set(data_labels))
    data_arr = []
    for l in labels:
      data_arr.append(np.array(arr1d)[np.array(data_labels) == l])
  else:
    data_arr = [np.array(arr1d)]
  for i, d in enumerate(data_arr):
    if data_labels is not None:
      l = labels[i]
    else:
      l = None
    fig.add_trace(go.Histogram(x=d, name=l))

  arg_dict = {'yaxis_title_text': 'Count', 'barmode': mode, 'xaxis_title_text': xaxis, 'title_text': title}
  fig.update_layout(**arg_dict)
  if mode == 'overlay':
    fig.update_traces(opacity=0.75)
  fig.show()