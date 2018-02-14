# -*- coding: utf-8 -*-
"""
Kaggle Competition - Transparent SemiConductors

Bandgap Energy Code - Contineud from Kaggle_Transparent_Semiconductors Code

"""

#################################################################

# Import the Libraries

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
validation_size = 0.20
seed = 1001

X_train, X_test, Y_train, Y_test = train_test_split(X, Y2, test_size = validation_size,
                                                    random_state = seed)
    

###################################################################

# Functions

###################################################################

def compare_algorithms(results):
    fig = pyplot.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    pyplot.boxplot(results)
    ax.set_xticklabels(names)
    pyplot.show()
    return

def KNN_Model(N_Clusters):
    scaler = StandardScaler().fit(X_train)
    rescaledX = scaler.transform(X_train)
    KNN_model = KNeighborsRegressor(n_neighbors = N_Clusters)
    KNN_model.fit(rescaledX, Y_train.ravel())
    rescaledtestX = scaler.transform(X_test)
    KNN_predictions = KNN_model.predict(rescaledtestX)
    print(mean_squared_error(Y_test.ravel(), KNN_predictions))
    return KNN_model

def GBM_Model(N_estimators):
    scaler = StandardScaler().fit(X_train)
    rescaledX = scaler.transform(X_train)
    GBM_model = GradientBoostingRegressor(random_state = seed, n_estimators = N_estimators)
    GBM_model.fit(rescaledX, Y_train.ravel())
    rescaledtestX = scaler.transform(X_test)
    predictions = GBM_model.predict(rescaledtestX)
    print(mean_squared_error(Y_test.ravel(), predictions))
    return GBM_model

def evaluate_model(model):
    scaler = StandardScaler().fit(X_train)
    rescaledX = scaler.transform(X_train)
    model.fit(rescaledX, Y_train.ravel())
    rescaledtestX = scaler.transform(X_test)
    predictions = model.predict(rescaledtestX)
    print(mean_squared_error(Y_test.ravel(), predictions))
    return    

def Test_Kaggle_Data(model):
    data_test = pd.read_csv("D:/Python/Datasets/Semiconductor_Data_Test.csv")
    data_test = data_test.drop(columns = ['id'])    
    cols_to_transform = ['spacegroup']
    data_test_2 = pd.get_dummies(data_test, columns = cols_to_transform )
    data_test_3 = data_test_2.copy()
    X_Kaggle_test = data_test_3.copy().values    
    rescaledKaggletestX = scaler.transform(X_Kaggle_test)
    predictions = model.predict(rescaledKaggletestX)
    Bandgap_Model_Predictions = pd.DataFrame(predictions)
    return Bandgap_Model_Predictions

def SVR_Tuning(kernel_values, C_values):
    scaler = StandardScaler().fit(X_train)
    rescaledX = scaler.transform(X_train)
    kernel = kernel_values
    C = C_values
    param_grid = dict(kernel=kernel, C = C)
    model = SVR
    kfold = KFold(n_splits=num_folds, random_state=seed)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
    grid_result = grid.fit(rescaledX, Y_train.ravel())
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    best_result = {}
    best_result[grid_result.best_score_] = grid_result.best_params_
    return best_result


###################################################################

# Model Building Phase

###################################################################

# Set the Variables

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

compare_algorithms(results)

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
k_values = np.array([i for i in range(1, 22, 1)])
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


# Test out a Model Using Gradient Boosting Model - Best so far is 150 (Function Defined Above)
    
GBM_Model_130 = GBM_Model(130)
GBM_Model_150 = GBM_Model(150)
GBM_Model_200 = GBM_Model(200)

GBM_Models = {}
for i in range(10, 500, 10):
    j = str(i) + " trees:"
    GBM_Models[j] =  GBM_Model(i)

print(GBM_Models)
    

# Test out a Model Using KNN with 7 Clusters (Function Defined Above)

for i in range(1, 20):
    print(str(i) + " nearest neighbors:")
    KNN_Model(i)

KNN_Model_5 = KNN_Model(5)
KNN_Model_7 = KNN_Model(7)
KNN_Model_9 = KNN_Model(9)
KNN_Model_13 = KNN_Model(13)
KNN_Model_15 = KNN_Model(15)

# Test out a model using Support Vector Machines - for Comparison - Function Defined Above (May need to be edited)

len(models)
models[5]
type(models[5])
SVR = models[5][1]
type(SVR)

kernel_values = ['rbf', 'poly', 'linear', 'sigmoid']
C_values = [i for i in range(1, 5, 1)]
SVR_Parameters_2 = SVR_Tuning(kernel_values, C_values)

SVR_2 = SVR(C=4.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',
            kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)

evaluate_model(SVR_2)


# Test this on Kaggle Test Data - Bandgap Energy 

Bandgap_Energy_KNN_5 = Test_Kaggle_Data(KNN_Model_5)
Bandgap_Energy_KNN_5.to_csv("Bandgap Energy KNN 5.csv")

Bandgap_Energy_KNN_9 = Test_Kaggle_Data(KNN_Model_9)
Bandgap_Energy_KNN_9.to_csv("Bandgap Energy KNN 9.csv")

Bandgap_Energy_SVR = Test_Kaggle_Data(SVR_2)
Bandgap_Energy_SVR.to_csv("Bandgap Energy SVR.csv")

Bandgap_Energy_KNN_15 = Test_Kaggle_Data(KNN_Model_15)
Bandgap_Energy_KNN_15.to_csv("Bandgap Energy KNN 15.csv")


