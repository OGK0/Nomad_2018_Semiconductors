# -*- coding: utf-8 -*-
"""
Kaggle Transparent SemiConductors - XGBoost 
Created on Sun Feb 11 18:46:55 2018

@author: khark
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

def Feature_Selection_Model(model):
    thresholds = sort(model.feature_importances_)
    for thresh in thresholds:
        selection = SelectFromModel(model, threshold=thresh, prefit=True)
        select_X_train = selection.transform(X_train)
        selection_model = XGBRegressor()
        selection_model.fit(select_X_train, Y_train)
        select_X_test = selection.transform(X_test)
        Y_pred = selection_model.predict(select_X_test)
        MSE_T = mean_squared_error(Y_test, Y_pred)
        print("Thresh=%.3f, n=%d, MSE: %.4f" % (thresh, select_X_train.shape[1], MSE_T))
    print("Done !")
    return
    
def Number_of_Trees_Grid_Search(min_trees, max_trees, inc):
    model = XGBRegressor()
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
    return

def Depth_of_Trees_Grid_Search(min_value, max_value, inc):
    model = XGBRegressor()
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
    return

def Tree_Number_and_Depth(num_of_trees_list, max_depth_list):
    model = XGBRegressor()
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
    return

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
    return

def Trees_and_Learning_Rate(tree_list, rate_list):
    model = XGBRegressor()
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
    return
        
def Sampling_Rates(subsample_list, colsample_list):
    model = XGBRegressor(n_estimators = 215, max_depth = 4, learning_rate = 0.1)
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
    return

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
    return

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
    return


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
    return


###################################################################

# XGBoost Model Building

###################################################################

"""
I will build and test the models on Y2, since this is where the maximum
improvement can be made to improve performance.

Benchmark - MSE of 0.056
"""

# Basic Model Building - Ch.4

xgb_1 = XGBRegressor()
xgb_1.fit(X_train, Y_train)
y_pred = xgb_1.predict(X_test)
predictions = [round(value) for value in y_pred]
MSE_1 = mean_squared_error(Y_test, predictions)
print("MSE is " + str(MSE_1))

plot_tree(xgb_1)

# Model Using KFold Cross Validation

xgb_2 = XGBRegressor()
kfold = KFold(n_splits = 10, random_state = 7)
results = cross_val_score(xgb_2, X_train, Y_train, cv = kfold)
xgb_2.fit(X_train, Y_train)
Y_pred = xgb_2.predict(X_test)
MSE_2 = mean_squared_error(Y_test, Y_pred)
print("MSE is: " + str(MSE_2))

feature_importances(xgb_2)

# Test Using Kaggle Data - XGB_2

Bandgap_XGB_2_Predictions = Test_Kaggle_Data(xgb_2)


# Using Feature Importances for XGB_2

thresholds = sort(xgb_2.feature_importances_)
for thresh in thresholds:
    selection = SelectFromModel(xgb_2, threshold=thresh, prefit=True)
    select_X_train = selection.transform(X_train)
    selection_model = XGBRegressor()
    selection_model.fit(select_X_train, Y_train)
    select_X_test = selection.transform(X_test)
    Y_pred = selection_model.predict(select_X_test)
    MSE_T = mean_squared_error(Y_test, Y_pred)
    print("Thresh=%.3f, n=%d, MSE: %.4f" % (thresh, select_X_train.shape[1], MSE_T))
print("Done !")


# XGB Model 3 

xgb_3 = XGBRegressor()

eval_set = [(X_train, Y_train), (X_test, Y_test)]
xgb_3.fit(X_train, Y_train,
          early_stopping_rounds = 5,
          eval_metric="rmse", 
          eval_set=eval_set, verbose=True)

Y_pred = xgb_3.predict(X_test)
MSE_3 = mean_squared_error(Y_test, Y_pred)
print("MSE is: " + str(MSE_3))

# plot log loss
fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
ax.plot(x_axis, results['validation_1']['rmse'], label='Test')
ax.legend()

# XGB Model 4 - Tuning the Number of Trees - Function Defined Above

Number_of_Trees_Grid_Search(50, 400, 50)

xgb_4 = XGBRegressor(n_estimators = 250)

eval_set = [(X_train, Y_train), (X_test, Y_test)]
xgb_4.fit(X_train, Y_train,
          early_stopping_rounds = 5,
          eval_metric="rmse", 
          eval_set=eval_set, verbose=True)

Y_pred = xgb_4.predict(X_test)
MSE_4 = mean_squared_error(Y_test, Y_pred)
print("MSE is: " + str(MSE_4))


# XGB Model 5 - Tuning the Depth of Trees - Function Defined Above

Depth_of_Trees_Grid_Search(1, 11, 1)

xgb_5 = XGBRegressor(n_estimators = 250, max_depth = 4)

eval_set = [(X_train, Y_train), (X_test, Y_test)]
xgb_5.fit(X_train, Y_train,
          early_stopping_rounds = 5,
          eval_metric="rmse", 
          eval_set=eval_set, verbose=True)

Y_pred = xgb_5.predict(X_test)
MSE_5 = mean_squared_error(Y_test, Y_pred)
print("MSE is: " + str(MSE_5))

# Test Using Kaggle Data - XGB_5

Bandgap_XGB_5_Predictions = Test_Kaggle_Data(xgb_5)
Bandgap_XGB_5_Predictions.to_csv("Bandgap Energy XGB 5 Predictions.csv")


# XGB Model 6 - Tuning the Number and Depth of Trees Simultaneously - Function Defined Above

Tree_Number_and_Depth([50, 100, 150, 200, 250, 300, 350, 400, 450, 500], 
                      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])


xgb_6 = XGBRegressor(n_estimators = 250, max_depth = 4)

eval_set = [(X_train, Y_train), (X_test, Y_test)]
xgb_6.fit(X_train, Y_train,
          early_stopping_rounds = 5,
          eval_metric="rmse", 
          eval_set=eval_set, verbose=True)

Y_pred = xgb_6.predict(X_test)
MSE_6 = mean_squared_error(Y_test, Y_pred)
print("MSE is: " + str(MSE_6))



# XGB Model 7 - Tuning the Learning Rate - Function Defined Above

Learning_Rate_Grid_Search([0.0001, 0.001, 0.01, 0.1, 0.2, 0.3])

Trees_and_Learning_Rate([50, 100, 150, 200, 215, 250, 300, 350, 400, 450, 500],
                        [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3])

Number_of_Trees_Grid_Search(200, 300, 5)

xgb_7 = XGBRegressor(n_estimators = 215, max_depth = 4, learning_rate = 0.1)

eval_set = [(X_train, Y_train), (X_test, Y_test)]
xgb_7.fit(X_train, Y_train,
          early_stopping_rounds = 5,
          eval_metric="rmse", 
          eval_set=eval_set, verbose=True)

Y_pred = xgb_7.predict(X_test)
MSE_7 = mean_squared_error(Y_test, Y_pred)
print("MSE is: " + str(MSE_7))

feature_importances(xgb_7)
Feature_Selection_Model(xgb_7)

# XGB Model 8 - Tuning the Other Variables

xgb_8 = XGBRegressor(n_estimators = 215, max_depth = 4, learning_rate = 0.1,
                     subsample = 0.7,
                     colsample_bytree = 0.7)

eval_set = [(X_train, Y_train), (X_test, Y_test)]
xgb_8.fit(X_train, Y_train,
          early_stopping_rounds = 5,
          eval_metric="rmse", 
          eval_set=eval_set, verbose=True)

Y_pred = xgb_8.predict(X_test)
MSE_8 = mean_squared_error(Y_test, Y_pred)
print("MSE is: " + str(MSE_8))

Feature_Selection_Model(xgb_8)



# XGB Model 9 - Subsample and Column Sample Rates - Function Defined Above

Sampling_Rates([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9],
               [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9])


xgb_9 = XGBRegressor(n_estimators = 215, max_depth = 4, learning_rate = 0.1,
                     subsample = 0.8,
                     colsample_bytree = 0.5)

evaluate_model(xgb_9)
Feature_Selection_Model(xgb_9)

# Yields MSE of 0.440

# Test Kaggle Data 

Bandgap_XGB_9_Predictions = Test_Kaggle_Data(xgb_9)
Bandgap_XGB_9_Predictions.to_csv("Bandgap Energy XGB_9 Predictions.csv")


# XGB 10 - Tuning Gamma

gamma_list = [i/20 for i in range(0,7)]
Gamma_Grid(xgb_9, gamma_list)

xgb_10 = XGBRegressor(n_estimators = 215, max_depth = 4, learning_rate = 0.1,
                     subsample = 0.8,
                     colsample_bytree = 0.5,
                     gamma = 0.2)

evaluate_model(xgb_10)
Bandgap_XGB_10_Predictions = Test_Kaggle_Data(xgb_10)
Bandgap_XGB_10_Predictions.to_csv("Bandgap Energy XGB_10 Predictions.csv")

# XGB 11 & 12 - Min_Child_Weight, Regularization Parameters

Min_Child_Weight_Grid(xgb_9, 1, 6, 1)

xgb_11 = XGBRegressor(n_estimators = 215, max_depth = 4, learning_rate = 0.1,
                     subsample = 0.8,
                     colsample_bytree = 0.5,
                     min_child_weight = 4)

evaluate_model(xgb_11)

alpha_list = [0, 0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.3, 0.5, 1]
Reg_Alpha_Grid(xgb_11, alpha_list)

xgb_12 = XGBRegressor(n_estimators = 215, max_depth = 4, learning_rate = 0.1,
                     subsample = 0.8,
                     colsample_bytree = 0.5,
                     min_child_weight = 4,
                     reg_alpha = 0.25,
                     gamma = 0.1)


# Test Kaggle Data for Model 11

Bandgap_XGB_11_Predictions = Test_Kaggle_Data(xgb_11)
Bandgap_XGB_11_Predictions.to_csv("Bandgap Energy XGB 11 Predictions.csv")













