# Classifiers.py
# ----------------------------------------------------------
# This is the classifiers module, has functions for hyperpar optimization /
# validation / ensembling (eventually)

import numpy as np
import pandas as pd
from time import time
import tqdm
import os
import multiprocessing
from functools import partial
cwd = os.path.abspath(os.path.curdir) 

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import roc_auc_score as auc
from sklearn.model_selection import GridSearchCV, KFold

np.random.seed(69)
n_cores = multiprocessing.cpu_count()
print(f'Number of cores to use: {n_cores}')


# Hyperparameter optimization funcs
# ----------------------------------------------------------

def SetupClassifiers(algs = ['RF', 'GDB', 'ADB']):
    ''' 
    This sets up the grids to grid search for a set of learning
    algorithms
    '''
    RF_parms = {'n_estimators':[20, 50, 100], 'max_features': ['sqrt', .5, 1], 
                'min_samples_leaf': [5,20,50]}
    #https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/
    ADB_parms = {'learning_rate':[.05, .1, .2], 'n_estimators':[20,50,80]} 
    GDB_parms = {'learning_rate':[.05, .1, .2], 'n_estimators':[20,50,80]}
     #Classifiers List to iterate through'''
    clsfr = {
            'ADB': (AdaBoostClassifier(), ADB_parms),
             'RF': (RandomForestClassifier(max_features = 0.5, min_samples_leaf = 5, n_estimators = 100), RF_parms),
             'GDB': (GradientBoostingClassifier(), GDB_parms) }
    classifiers = [(k,) + v for k, v in clsfr.items() if k in algs] 
    return classifiers


def TuneClassifiers(data_train, data_test, labels_train, labels_test, algs=['RF', 'GDB', 'ADB']):
    '''
    This runs grid search to find the best parameters for models

    Args:
        -data_train/test: train/testing data as 2D np.arrays
        -labels_train/test: binary (0/1) labels for training/testing data
        -algs: which algs to grid search

    Returns:
        -results_train: pd.DataFrame of training results
        -results_test: pd.DataFrame of testing results
        -all_models: a dictionary mapping algorithm name to trained 
                gridsearch model (so can extract model.best_params_ and
                stuff later)
    '''
    classifiers = SetupClassifiers(algs)
    cols = ['Classifier', 'Type', 'Tuning', 'Accuracy', 'AUC']
    res_shape = (2 * len(classifiers), len(cols))
    res = 0
    tune_results = pd.DataFrame(columns=cols, data=np.empty(res_shape))
     
    # run the base (non-tuned) models
    for c, func, parameters in classifiers:
        model = func 
        model.fit(data_train,labels_train)
        pred_train = model.predict(data_train)
        pred_test = model.predict(data_test)
        
        acc_train = ((pred_train == labels_train).mean()) 
        acc_test = ((pred_test == labels_test).mean())

        auc_train = auc(labels_train, model.predict_proba(data_train)[:, 1])
        auc_test = auc(labels_test, model.predict_proba(data_test)[:, 1])

        tune_results.loc[res] = [c, 'train', 'base', acc_train, auc_train]
        tune_results.loc[res+1] = [c, 'test', 'base', acc_test, auc_test]
        res = res+2
    
    # run model tuning
    
    all_models = {}
    for c, func, parameters in classifiers:
        t0 = time()
        print("Tuning", c, "...")
        
        tun_model = GridSearchCV(func, parameters, n_jobs=n_cores, cv=3, iid=True) 
        tun_model.fit(data_train,labels_train)
        
        t1 = time()
        print("Tuned in:", t1-t0)
        
        print(tun_model.best_params_)
        all_models[c] = tun_model

        pred_train = tun_model.predict_proba(data_train)[:, 1]
        pred_test = tun_model.predict_proba(data_test)[:, 1]
        
        auc_train = auc(labels_train, pred_train)
        auc_test = auc(labels_test, pred_test)

        acc_train = ((np.round(pred_train) == labels_train).mean()) 
        acc_test = ((np.round(pred_test) == labels_test).mean())

        tune_results.loc[res] = [c, 'train', 'base', acc_train, auc_train]
        tune_results.loc[res+1] = [c, 'test', 'tuned', acc_test, auc_test]
        res = res+2 
    
    #tune_results.to_csv('tuning_results.txt', sep=' ')  
    
    results_train = tune_results.loc[(tune_results['Type'] == 'train')]
    results_test = tune_results.loc[(tune_results['Type'] == 'test')]
    
    return results_train, results_test, all_models



# Hyperparameter optimization funcs
# ----------------------------------------------------------

def run_fold(inp, model, X0, Y0, get_pars, process):
    '''Helper function for multiprocessing in cross_val'''
    train_inds, val_inds = inp
    X, Y = X0[train_inds], Y0[train_inds]
    valX, valY = X0[val_inds], Y0[val_inds]
    pars = get_pars(X)
    X, valX = process(X, pars), process(valX, pars)
    model.fit(X, Y)
    preds = model.predict_proba(valX)[:, 1]
    return auc(valY, preds), (np.round(preds) == valY).mean()

def cross_val(model, X0, Y0, process_func_pair, k=3, verbose=True):
    '''
    This function performs cross_validation (kfold) on a model

    Args:
        -model: model to CV
        -X0, Y0: data
        -process_func_pair: a 2-tuple (get_pars, process) where
            get_pars will be called on train data to get params to
            pass to process, which will process training and test data
        -k: number of folds
        -verbose: bool flag, set to T to get tqdm progress bar

    Returns:
        -aucs: List of aucs per fold
        -accs: list of accuracies per fold
    '''

    pool = multiprocessing.Pool(n_cores)
    get_pars, process = process_func_pair
    kf = KFold(n_splits=k, shuffle=True)

    run_fold_w_args = partial(run_fold, model=model, X0=X0, Y0=Y0, get_pars=get_pars, process=process)
    if verbose:
        res = list(tqdm.tqdm(pool.imap(run_fold_w_args, kf.split(X0)), desc='CV fold'))
    else:
        res = list(pool.imap(run_fold_w_args, kf.split(X0)), desc='CV fold')

    aucs, accs = zip(*res)
    return np.array(aucs), np.array(accs)


