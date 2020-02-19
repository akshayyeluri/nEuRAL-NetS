# nEuRAL NetS
CS155 Caltech 2020 Kaggle Repo for team nEuRAL NetS

Team members:
1. Akshay Yeluri
2. Rashida Hakim
3. Nicholas Chang

## Files:
classifiers.py: Cross validation and model setup / selection for models used in this kaggle

transforms.py: Functions for modifying / transforming data prior to training / testing

\*.pkl: Saved models

\*.json: grid search for hyperparameters results. 

## Notebooks:
clsf.ipynb: Main notebook, reads in data, transforms it, sets up models / hyperparameters to grid search, 
            does kfold CV, trains on all train data, gets predictions on test data
            
ens.ipynb: Ensemble models, allows for easy ensembling




