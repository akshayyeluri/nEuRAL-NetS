# Transforms.py
# ----------------------------------------------------------
# This is the transforms module, has functions for transforming
# the data. Transforms are in two flavors
#
# 1) Feature engineering (see transform df), 
# which shuffles around features to get
# better features to train on
#
# 2) data correction / normalization (see process and get_pars)
# This fixes data (column normalization, nan removal, etc.)

import numpy as np
import pandas as pd
import re

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold

np.random.seed(69)


# Feature engineering
# ----------------------------------------------------------

# New version that finds group inds
def transform_df(df0, train=False, as_df=False, **kwargs):
    '''
    This function feature engineers an initial dataframe

    Args:
        -df0: a pd.DataFrame instance to engineer
        -train: boolean flag, set to True to return old / new labels
        -as_df: boolean flag, if False will return just a numpy array
                of data, rather than a pd.DataFrame

    Returns:
        -X: either a pd.DataFrame (if as_df=True) or np array of modified
            / feature engineered data
        -oldY: only returned if train=True, these are original labels
        -newY: modified (maybe regression, if doing Rashida's thing)
               labels, also returned only if train=True
        -group_inds: list of lists for inds of cols that must be normalized
               together
    '''

    df = df0.copy()
    drop_cols = ['y'] if train else []
    df = df.drop(drop_cols, axis = 1)
    
    vol_inds = [i for i, col in enumerate(df.columns) if 'vol' in col] 
    price_cols = ['mid', 'last_price'] + \
                 [f'bid{i}' for i in range(1,6)] + \
                 [f'ask{i}' for i in range(1,6)]
    price_inds = [i for i, col in enumerate(df.columns) if col in price_cols] 
    group_inds = [vol_inds, price_inds]

    X = df if as_df else df.values
    if train:
        # Get regression y's
        # newY = np.concatenate(((df0['mid'].values[2:] - df0['mid'].values[:-2]), df0['y'].values[-2:]))
        
        # Get smooth label y's
        k = kwargs.get('k', 20) # Window size for smoothing
        newY = df0['y'].values.copy()
        if k:
            newY[k:-k] = smooth_labels(df0['mid'].values, k=k)
        return X, df0['y'].values, newY, group_inds
    return X



# Old version that actually does shit
#def transform_df(df0, train=False, as_df=False, **kwargs):
#    '''
#    This function feature engineers an initial dataframe
#
#    Args:
#        -df0: a pd.DataFrame instance to engineer
#        -train: boolean flag, set to True to return old / new labels
#        -as_df: boolean flag, if False will return just a numpy array
#                of data, rather than a pd.DataFrame
#
#    Returns:
#        -X: either a pd.DataFrame (if as_df=True) or np array of modified
#            / feature engineered data
#        -oldY: only returned if train=True, these are original labels
#        -newY: modified (maybe regression, if doing Rashida's thing)
#               labels, also returned only if train=True
#    '''
#
#    df = df0.copy()
#    drop_cols = ['y'] if train else []
#        
#    #bid_diffs = pd.DataFrame({f'bid_w{i}': ((df[f'bid{i}'] - df['mid']) * df[f'bid{i}vol']) \
#    #                          for i in range(1, 6)})
#    #ask_diffs = pd.DataFrame({f'ask_w{i}': ((df[f'ask{i}'] - df['mid']) * df[f'ask{i}vol']) \
#    #                          for i in range(1, 6)})
#    #drop_cols += [f'{a}{b}{c}' for a in ['bid', 'ask'] for b in range(1, 6) for c in ["", "vol"]] + \
#    #             ['id', 'mid']
#    #df = df.drop(drop_cols, axis = 1).join(bid_diffs).join(ask_diffs)
#    df = df.drop(drop_cols, axis = 1)
#    X = df if as_df else df.values
#    if train:
#        # Get regression y's
#        # newY = np.concatenate(((df0['mid'].values[2:] - df0['mid'].values[:-2]), df0['y'].values[-2:]))
#        
#        # Get smooth label y's
#        k = kwargs.get('k', 20) # Window size for smoothing
#        newY = df0['y'].values.copy()
#        if k:
#            newY[k:-k] = smooth_labels(df0['mid'].values, k=k)
#        return X, df0['y'].values, newY
#    return X



def smooth_labels(mids, k=20, alpha=1.0):
    '''
    This function gets smooth labels given mid prices

    Args:
        -mids: np.array of mid prices
        -k: window size for mean smoothing
        -alpha: smoothing parameter, controls how much bigger than smallest
                magnitude jump a jump has to be to get 1 label

    Returns:
        -q_smooth: smooth labels, np.array of size (N - 2k,), where
                   N = mids.size
    '''
    q = mids
    q1 = np.diff(q)
    thresh = np.min(np.absolute(q1[np.nonzero(q1)])) * alpha
    q_smooth = np.array([q[i+k:i+2*k].mean() - q[i:i+k].mean() >= thresh for i in \
                      range(q.shape[0] - 2 * k)])
    return q_smooth



# Data processing / Normalization
# ----------------------------------------------------------
def get_pars_for_processing(X, group_inds = []):
    '''
    This function gets parameters from training data to process the training
    and testing data (passing these parameters as args to process_with_pars)

    Args:
        -X: training data to get pars from

    Returns:
        -pars: pars for processing
    '''
    scaler = StandardScaler()
    scaler.fit(X)
    for group in group_inds:
        inds = np.array(group)
        scaler.mean_[inds] = (scaler.mean_[inds]).mean()
        scaler.var_[inds] = (scaler.var_[inds]).mean()
        scaler.scale_ = np.sqrt(scaler.var_)
    return scaler

def process_with_pars(X, params):
    '''
    Function that takes training / test data, 
    and process it for training / evaluation
    '''
    scaler = params
    where_nan = np.isnan(X)
    scaled = scaler.transform(X)
    X[where_nan] = 0
    return X


# Output transform
# ----------------------------------------------------------
def scale(vals):
    return (vals - vals.min()) / np.ptp(vals)
