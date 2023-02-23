import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from alepython.ale import _first_order_ale_quant
from alepython.ale import _second_order_ale_quant,_second_order_quant_plot,_get_centres
from scipy.interpolate import interp2d
from matplotlib.patches import Rectangle

from typing import Callable,Iterable,Union

def first_order_aleplot_quant(
    predictor:Callable,
    train_data:pd.DataFrame,
    feature:str,
    ax:matplotlib.axes.Axes,
    bins:int=10,
    **kwargs):
    '''
    Plot and return the first-order ALE function for a continuous feature.

    Arguments
    ---------
    predictor: callable
        The prediction function. For scikit-learn regressors, pass the `.predict`
        method. For scikit-learn classifiers, either pass the `.predict_proba` method
        or pass a custom log-odds function
    
    train_data: pd.DataFrame
        Training data on which the model was trained. Cannot pass numpy ndarrays.
    
    feature: str
        Feature name. A single column label
    
    ax: matplotlib.axes.Axes
        Pre-existing axes to plot onto
    
    bins : int
        This defines the number of bins to compute. The effective number of bins may
        be less than this as only unique quantile values of train_data[feature] are
        used.
    
    **kwargs: plot properties, optional
        Additional keyword parameters passed to `ax.plot`.
    
    Returns
    ---------
    ax: matplotlib.axes.Axes
        The matplotlib axes containing the plot
    
    ale: np.ndarray
        first order ALE
    
    quantiles: np.ndarray
        The quantiles used
    '''

    ale,quantiles = _first_order_ale_quant(predictor,train_data,feature,bins)
    bin_centers = 0.5*(quantiles[:-1]+quantiles[1:])
    _ = ax.plot(bin_centers,ale,**kwargs)
    _ = sns.rugplot(train_data[feature],ax=ax,color='k',alpha=0.2)
    _ = ax.grid(linestyle="-", alpha=0.4)
    _ = ax.set_xlabel(feature)
    _ = ax.set_ylabel(r'$f_1$(%s)'%feature)
    return ax,ale,quantiles


def second_order_aleplot_quant(
    predictor:Callable,
    train_data:pd.DataFrame,
    features:Iterable[str],
    fig:matplotlib.figure.Figure,
    ax:matplotlib.axes.Axes,
    bins:Union[int,Iterable[int]]=10,
    combined:bool=False,
    mark_empty:bool=True,
    **kwargs):
    '''
    Plot and return the first-order ALE function for a continuous feature.

    Arguments
    ---------
    predictor: callable
        The prediction function. For scikit-learn regressors, pass the `.predict`
        method. For scikit-learn classifiers, either pass the `.predict_proba` method
        or pass a custom log-odds function
    
    train_data: pd.DataFrame
        Training data on which the model was trained. Cannot pass numpy ndarrays.
    
    features: 2-iterable[str]
        The names of the two features
    
    fig: matplotlib.Figure
        pre-existing matplotlib figure object

    ax: matplotlib.axes.Axes
        Pre-existing axes to plot onto
    
    bins : 2-iterable[int] or int (Default: 10)
        This defines the number of bins to compute. The effective number of bins may
        be less than this as only unique quantile values of train_data[feature] are
        used. If only one integer is given, this is used for both the features.

    combined: bool (Default:False)
        If True, the first-order effects are also included in the plot
    
    mark_empty: bool (Default:True)
        If True, plot rectangles over bins that did not contain any samples.
    
    **kwargs: plot properties, optional
        Additional keyword parameters passed to `ax.contourf`.
    
    Returns
    ---------
    ax: matplotlib.axes.Axes
        The matplotlib axes containing the plot
    '''
    if isinstance(bins,int):
        bins = [bins]*2

    ale,quantiles_list = _second_order_ale_quant(predictor,train_data,features,bins)
    centres_list = [_get_centres(quantiles) for quantiles in quantiles_list]
    ale_interp = interp2d(centres_list[0], centres_list[1], ale.T)

    second_order = ale_interp(centres_list[0],centres_list[1])
    if combined:
        # add first order ALE
        ale_x,_= _first_order_ale_quant(predictor,train_data,features[0],bins=bins[0])
        ale_y,_= _first_order_ale_quant(predictor,train_data,features[1],bins=bins[1])

        second_order += ale_x.reshape(1,-1).repeat(ale_y.shape[0],axis=0) + \
            ale_y.reshape(-1,1).repeat(ale_x.shape[0],axis=1)

    X,Y = np.meshgrid(centres_list[0], centres_list[1], indexing="xy")
    CF = ax.contourf(X, Y, second_order, **kwargs)
    if mark_empty and np.any(ale.mask):
        # Do not autoscale, so that boxes at the edges (contourf only plots the bin
        # centres, not their edges) don't enlarge the plot.
        plt.autoscale(False)
        # Add rectangles to indicate cells without samples.
        for i, j in zip(*np.where(ale.mask)):
            _=ax.add_patch(
                Rectangle(
                    [quantiles_list[0][i], quantiles_list[1][j]],
                    quantiles_list[0][i + 1] - quantiles_list[0][i],
                    quantiles_list[1][j + 1] - quantiles_list[1][j],
                    linewidth=1,
                    edgecolor="k",
                    facecolor="none",
                    alpha=0.4,
                )
            )

    _ = fig.colorbar(CF)
    _ = ax.set_xlabel(features[0])
    _ = ax.set_ylabel(features[1])
    return ax
