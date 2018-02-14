# -*- coding: utf-8 -*-
"""
Kaggle Transparent SemiConductors - XGBoost  - Tuning Models
"""

#################################################################

# Import the Libraries

#################################################################

import numpy as np
from numpy import sort
import pandas as pd
from pandas import set_option
from pandas.plotting import scatter_matrix

import xgboost as xgb
from xgboost import XGBRegressor
from xgboost import plot_tree
from xgboost import plot_importance
from matplotlib import pyplot
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import accuracy_score

# View XGBoost Trees - Run this only if needed

"""
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
"""

###################################################################

# Load the Data and Prepare it

###################################################################

data_1 = pd.read_csv("D:/Python/Datasets/Semiconductor_Data_1.csv")
type(data_1)
data_1 = data_1.drop(columns = ['id'])

cols_to_transform = ['spacegroup']
data_2 = pd.get_dummies(data_1, columns = cols_to_transform )

data_3 = data_2.copy()
X = data_3.drop(['formation_energy_ev_natom', 'bandgap_energy_ev'], axis = 1).values
Y1 = data_3[['formation_energy_ev_natom']].values
Y2 = data_3[['bandgap_energy_ev']].values
validation_size = 0.33
seed = 1001

X_train, X_test, Y_train, Y_test = train_test_split(X, Y2, test_size = validation_size,
                                                    random_state = seed)

# There is no missing data in this dataset.

###################################################################

# Functions

###################################################################

def feature_importances(model):
    pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
    pyplot.show()
    plot_importance(model)
    pyplot.show()
    return

def evaluate_model(model):
    eval_set = [(X_train, Y_train), (X_test, Y_test)]
    model.fit(X_train, Y_train,
          early_stopping_rounds = 5,
          eval_metric="rmse", 
          eval_set=eval_set, verbose=True)
    Y_pred = model.predict(X_test)
    MSE_model = mean_squared_error(Y_test, Y_pred)
    print("MSE is: " + str(MSE_model))
    return

def Test_Kaggle_Data(model):
    data_test = pd.read_csv("D:/Python/Datasets/Semiconductor_Data_Test.csv")
    data_test = data_test.drop(columns = ['id'])
    cols_to_transform = ['spacegroup']
    data_test_2 = pd.get_dummies(data_test, columns = cols_to_transform )
    data_test_3 = data_test_2.copy()
    X_Kaggle_test = data_test_3.copy().values
    Predictions = model.predict(X_Kaggle_test)
    Bandgap_Predictions = pd.DataFrame(Predictions)
    return Bandgap_Predictions

def Number_of_Trees_Grid_Search(input_model, min_trees, max_trees, inc):
    model = input_model
    n_estimators = range(min_trees, max_trees, inc)
    param_grid = dict(n_estimators=n_estimators)
    kfold = KFold(n_splits=10, shuffle=True, random_state=7)
    grid_search = GridSearchCV(model, param_grid, scoring="neg_mean_squared_error", n_jobs=-1, cv=kfold)
    grid_result = grid_search.fit(X_train, Y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    return grid_result.best_score_, grid_result.best_params_

def Depth_of_Trees_Grid_Search(input_model, min_value, max_value, inc):
    model = input_model
    max_depth = range(min_value, max_value, inc)
    print(max_depth)
    param_grid = dict(max_depth=max_depth)
    kfold = KFold(n_splits=10, shuffle=True, random_state=7)
    grid_search = GridSearchCV(model, param_grid, scoring="neg_mean_squared_error", n_jobs=-1, cv=kfold,
    verbose=1)
    grid_result = grid_search.fit(X_train, Y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    return grid_result.best_score_, grid_result.best_params_

def Tree_Number_and_Depth(input_model, num_of_trees_list, max_depth_list):
    model = input_model
    n_estimators = num_of_trees_list
    max_depth = max_depth_list
    print(max_depth)
    param_grid = dict(max_depth=max_depth, n_estimators=n_estimators)
    kfold = KFold(n_splits=10, shuffle=True, random_state=7)
    grid_search = GridSearchCV(model, param_grid, scoring="neg_mean_squared_error", n_jobs=-1, cv=kfold,
    verbose=1)
    grid_result = grid_search.fit(X_train, Y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    return grid_result.best_score_, grid_result.best_params_

def Learning_Rate_Grid_Search(input_model, rate_list):
    model = input_model
    learning_rate = rate_list
    param_grid = dict(learning_rate=learning_rate)
    kfold = KFold(n_splits=10, shuffle=True, random_state=7)
    grid_search = GridSearchCV(model, param_grid, scoring="neg_mean_squared_error", n_jobs=-1, cv=kfold)
    grid_result = grid_search.fit(X_train, Y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    return grid_result.best_score_, grid_result.best_params_

def Trees_and_Learning_Rate(input_model, tree_list, rate_list):
    model = input_model
    n_estimators = tree_list
    learning_rate = rate_list
    param_grid = dict(learning_rate=learning_rate, n_estimators=n_estimators)
    kfold = KFold(n_splits=10, shuffle=True, random_state=7)
    grid_search = GridSearchCV(model, param_grid, scoring="neg_mean_squared_error", n_jobs=-1, cv=kfold)
    grid_result = grid_search.fit(X_train, Y_train)    
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    return grid_result.best_score_, grid_result.best_params_
        
def Sampling_Rates(input_model, subsample_list, colsample_list):
    model = input_model
    subsample = subsample_list
    colsample_bytree = colsample_list
    param_grid = dict(subsample=subsample, colsample_bytree = colsample_bytree)
    kfold = KFold(n_splits=10, shuffle=True, random_state=7)
    grid_search = GridSearchCV(model, param_grid, scoring="neg_mean_squared_error", n_jobs=-1, cv=kfold)
    grid_result = grid_search.fit(X_train, Y_train)    
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    return grid_result.best_score_, grid_result.best_params_

def Gamma_Grid(input_model, gamma_list):
    model = input_model
    gamma = gamma_list
    param_grid = dict(gamma = gamma)
    kfold = KFold(n_splits=10, shuffle=True, random_state=7)
    grid_search = GridSearchCV(model, param_grid, scoring="neg_mean_squared_error", n_jobs=-1, cv=kfold)
    grid_result = grid_search.fit(X_train, Y_train)    
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    return grid_result.best_score_, grid_result.best_params_

def Min_Child_Weight_Grid(input_model, min_weight, max_weight, inc):
    model = input_model
    min_child_weight = range(min_weight, max_weight, inc)
    param_grid = dict(min_child_weight = min_child_weight)
    kfold = KFold(n_splits=10, shuffle=True, random_state=7)
    grid_search = GridSearchCV(model, param_grid, scoring="neg_mean_squared_error", n_jobs=-1, cv=kfold)
    grid_result = grid_search.fit(X_train, Y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    return grid_result.best_score_, grid_result.best_params_


def Reg_Alpha_Grid(input_model, alpha_list):
    model = input_model
    reg_alpha = alpha_list
    param_grid = dict(reg_alpha = reg_alpha)
    kfold = KFold(n_splits=10, shuffle=True, random_state=7)
    grid_search = GridSearchCV(model, param_grid, scoring="neg_mean_squared_error", n_jobs=-1, cv=kfold)
    grid_result = grid_search.fit(X_train, Y_train)    
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    return grid_result.best_score_, grid_result.best_params_


###################################################################

# XGBoost Model Building

###################################################################

"""
I will build and test the models on Y2, since this is where the maximum
improvement can be made to improve performance.

Benchmark - MSE of 0.056
"""

# Best Model So Far - Model 11 - Focus on Tuning the Model

xgb_11 = XGBRegressor(n_estimators = 215, max_depth = 4, learning_rate = 0.1,
                     subsample = 0.8,
                     colsample_bytree = 0.5,
                     min_child_weight = 4)

evaluate_model(xgb_11)


# Tuning Alpha with the Model

xgb_12 = XGBRegressor(n_estimators = 215, max_depth = 4, learning_rate = 0.1,
                     subsample = 0.8,
                     colsample_bytree = 0.5,
                     min_child_weight = 4,
                     reg_alpha = 0.66)

evaluate_model(xgb_12)

# Tuning the Child Weight with the Model

Min_Child_Weight_Grid(xgb_12, 1, 6, 1)

xgb_12 = XGBRegressor(n_estimators = 215, max_depth = 4, learning_rate = 0.1,
                     subsample = 0.8,
                     colsample_bytree = 0.5,
                     min_child_weight = 4,
                     reg_alpha = 0.66)

evaluate_model(xgb_12)

# Tuning the Other Parameters

tree_list = [i for i in range(10, 400, 10)]
depth_list = [i for i in range(1, 16, 1)]
Score, Parameters = Tree_Number_and_Depth(xgb_12, tree_list, depth_list)

xgb_13 = XGBRegressor(n_estimators = 260, max_depth = 3, learning_rate = 0.1,
                     subsample = 0.8,
                     colsample_bytree = 0.5,
                     min_child_weight = 4,
                     reg_alpha = 0.66)

evaluate_model(xgb_13)

# Model 13 is inferior to 12

learning_rate_list = [(i/100) * 2  for i in range(1, 50, 1)]
Score, Parameters = Trees_and_Learning_Rate(xgb_12, tree_list, learning_rate_list)

xgb_14 = XGBRegressor(n_estimators = 150, max_depth = 4, learning_rate = 0.1,
                     subsample = 0.8,
                     colsample_bytree = 0.5,
                     min_child_weight = 4,
                     reg_alpha = 0.66)

evaluate_model(xgb_14)

# Model 14 = Model 12

sampling_rates = [(i/100) for i in range(50, 100, 5)]
Score, Parameters = Sampling_Rates(xgb_14, sampling_rates, sampling_rates)


xgb_15 = XGBRegressor(n_estimators = 150, max_depth = 4, learning_rate = 0.1,
                     subsample = 0.85,
                     colsample_bytree = 0.5,
                     min_child_weight = 4,
                     reg_alpha = 0.66)

evaluate_model(xgb_15)

# Model 15 is worse than 14

gamma_list = [i/20 for i in range(0,10)]
Score, Parameters = Gamma_Grid(xgb_14, gamma_list)

xgb_16 = XGBRegressor(n_estimators = 150, max_depth = 4, learning_rate = 0.1,
                     subsample = 0.85,
                     colsample_bytree = 0.5,
                     min_child_weight = 4,
                     reg_alpha = 0.66,
                     gamma = 0.15)

evaluate_model(xgb_16)

alpha_list = [i/100 for i in range(1,105,5)]
Score, Parameters = Reg_Alpha_Grid(xgb_12, alpha_list)


# Test Kaggle Data on Model 12

Bandgap_Energy_XGB_12 = Test_Kaggle_Data(xgb_12)
Bandgap_Energy_XGB_12.to_csv("Bandgap Energy XGB_12.csv")


