import pandas as pd
import numpy as np
import plotly.graph_objects as go
#import plotly.figure_factory as ff


def divide_data(arr1d, data_labels=None):
    if data_labels is not None:
        labels = sorted(set(data_labels))
        data_arr = []
        for l in labels:
            data_arr.append(np.array(arr1d)[np.array(data_labels) == l])
        labels = list(map(lambda x: str(x), labels))
    else:
        labels = None
        data_arr = [np.array(arr1d)]
    return data_arr, labels


## mode in ['overlay', 'stack']
def plot_histogram(arr1d, mode='stack', data_labels=None, xaxis=None, title=None):
    data_arr, labels = divide_data(arr1d, data_labels) 
    fig = go.Figure()
    for i, d in enumerate(data_arr):
        if labels is not None:
            l = labels[i]
        else:
            l = None
        fig.add_trace(go.Histogram(x=d, name=l))

    arg_dict = {'yaxis_title_text': 'Count', 'barmode': mode, 'xaxis_title_text': xaxis, 'title_text': title}
    fig.update_layout(**arg_dict)
    if mode == 'overlay':
        fig.update_traces(opacity=0.75)
    fig.show()


def plot_box(arr1d, axis, data_labels, axis_title, no_bg=False):
    data_arr, labels = divide_data(arr1d, data_labels)
    fig = go.Figure()
    for i, l in enumerate(labels):
        if axis == 'x':
            fig.add_trace(go.Box(x=data_arr[i], name=l))
        elif axis == 'y':
            fig.add_trace(go.Box(y=data_arr[i], name=l))
    
    if axis == 'x':
        fig.update_layout(xaxis_title=axis_title)
    elif axis == 'y':
        fig.update_layout(yaxis_title=axis_title)
    
    if no_bg:
        fig.layout.plot_bgcolor = '#fff'
    fig.show()
