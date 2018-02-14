# -*- coding: utf-8 -*-
"""
Kaggle Competition - Nomad2018 Predicting Transparent Conductors
Code for Formation Group Energies

Created on Sat Feb 10 17:26:38 2018

@author: khark
"""

#################################################################

# Load the Libraries

#################################################################

import numpy as np
import pandas as pd
from pandas import set_option
from pandas.plotting import scatter_matrix
import xgboost as xgb
import tensorflow as tf
from matplotlib import pyplot
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
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

#################################################################

# Import the Data

#################################################################

data_1 = pd.read_csv("D:/Python/Datasets/Semiconductor_Data_1.csv")
type(data_1)
data_1 = data_1.drop(columns = ['id'])

#################################################################

# Functions

#################################################################

def read_data():
    data = pd.read_csv("D:/Python/Datasets/Semiconductor_Data_1.csv")
    data = data.drop(columns = ['id'])
    return data

def exploratory_analysis(data):
    print("The dimensions of the data are: " + str(data.shape))
    print(data.dtypes)

def correlation_matrix_plot(data):
    names = list(data.columns.values)
    fig = pyplot.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(data.corr(), vmin=-1, vmax=1, interpolation='none')
    fig.colorbar(cax)
    ticks = np.arange(0,14,1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    pyplot.show()
    return

def One_Hot_Encode(cols_to_transform, df):
    df_with_dummies = pd.get_dummies(df, columns = cols_to_transform )
    return df_with_dummies
    

#################################################################

# Executing the Code

#################################################################

# Descriptive Statistics
    
print(data_1.shape)
print(data_1.dtypes)
print(data_1.head(20))
print(data_1.corr(method = 'pearson'))

correlations = data_1.corr(method = 'pearson')
correlations.to_csv("Variable_Correlations.csv")

data_1.hist(sharex = False, sharey = False, xlabelsize = 1, ylabelsize = 1)
pyplot.show()

data_1.plot(kind='box', subplots=True, layout=(4,4), sharex=False, sharey=False,
            fontsize=8)
pyplot.show()

data_1.plot(kind='density', subplots=True, layout=(4,4), sharex=False, legend=False,
            fontsize=1)
pyplot.show()

scatter_matrix(data_1)
pyplot.show()

# Correlation matrix
correlation_matrix_plot(data_1)

# Encode the Categorical Variables - Only 1 here - Spacegroup

cols_to_transform = ['spacegroup']
data_2 = pd.get_dummies(data_1, columns = cols_to_transform )

# This file does not have any missing values that need to be dealt with

# Prepare the X and Y Datasets

data_3 = data_2.copy()
X = data_3.drop(['formation_energy_ev_natom', 'bandgap_energy_ev'], axis = 1).values
Y1 = data_3[['formation_energy_ev_natom']].values
Y2 = data_3[['bandgap_energy_ev']].values
validation_size = 0.20
seed = 1001

X_train, X_test, Y_train, Y_test = train_test_split(X, Y1, test_size = validation_size,
                                                    random_state = seed)

# Evaluate a Set of Algorithms

num_folds = 10
seed = 1001
scoring = 'neg_mean_squared_error'

models = []
models.append(('LR', LinearRegression()))
models.append(('LASSO', Lasso()))
models.append(('EN', ElasticNet()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('SVR', SVR()))

results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train.ravel(), cv=kfold, 
                                 scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Compare Algorithms

fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()


# Algorithms with Standardization Algorithms

pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR',
                                        LinearRegression())])))
pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()),('LASSO',
                                           Lasso())])))
pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()),('EN',
                                        ElasticNet())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN',
                                         KNeighborsRegressor())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART',
                                          DecisionTreeRegressor())])))
pipelines.append(('ScaledSVR', Pipeline([('Scaler', StandardScaler()),('SVR', SVR())])))

results = []
names = []
for name, model in pipelines:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train.ravel(), cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    
# Compare Algorithms for Pipeline and Scaling

fig = pyplot.figure()
fig.suptitle('Scaled Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

# KNN Algorithm tuning
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
k_values = np.array([1,3,5,7,9,11,13,15,17,19,21])
param_grid = dict(n_neighbors=k_values)
model = KNeighborsRegressor()
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train.ravel())

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# Ensemble Models
    
ensembles = []
ensembles.append(('ScaledAB', Pipeline([('Scaler', StandardScaler()),('AB',
                                        AdaBoostRegressor())])))
ensembles.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()),('GBM',
                                         GradientBoostingRegressor())])))
ensembles.append(('ScaledRF', Pipeline([('Scaler', StandardScaler()),('RF',
                                        RandomForestRegressor())])))
ensembles.append(('ScaledET', Pipeline([('Scaler', StandardScaler()),('ET',
                                        ExtraTreesRegressor())])))

results = []
names = []
for name, model in ensembles:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train.ravel(), cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Compare Boosting and Bagging Algorithms
    
fig = pyplot.figure()
fig.suptitle('Scaled Ensemble Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

# Gradient Boosting Estimator Model

scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
param_grid = dict(n_estimators = np.array([50,100,150,200,250,300,350,400]))
model = GradientBoostingRegressor(random_state=seed)
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train.ravel())

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# Test out a Model Using Gradient Boosting Model - Best so far is 150

scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = GradientBoostingRegressor(random_state=seed, n_estimators = 150)
model.fit(rescaledX, Y_train.ravel())

rescaledtestX = scaler.transform(X_test)
predictions = model.predict(rescaledtestX)
print(mean_squared_error(Y_test.ravel(), predictions))


# Test this on Kaggle Test Data - Formation Energy - GBM Model

data_test = pd.read_csv("D:/Python/Datasets/Semiconductor_Data_Test.csv")
type(data_test)
data_test = data_test.drop(columns = ['id'])

cols_to_transform = ['spacegroup']
data_test_2 = pd.get_dummies(data_test, columns = cols_to_transform )

data_test_3 = data_test_2.copy()
X_Kaggle_test = data_test_3.copy().values

rescaledKaggletestX = scaler.transform(X_Kaggle_test)
predictions = model.predict(rescaledKaggletestX)
type(predictions)
Formation_GBM_Predictions = pd.DataFrame(predictions)
Formation_GBM_Predictions.to_csv("Formation Energy Predictions.csv")











