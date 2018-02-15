# -*- coding: utf-8 -*-
"""
Kaggle Competition - Transparent SemiConductors - Ensemble Models
Created on Wed Feb 14 18:32:59 2018


Bechmark: 0.0564
Let's try to combine models to see if
the combined model has greater predictive power

"""

###################################################################

# Libraries

###################################################################

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

def xgb_feature_importances(model):
    pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
    pyplot.show()
    plot_importance(model)
    pyplot.show()
    return

def evaluate_model(model):
    scaler = StandardScaler().fit(X_train)
    rescaledX = scaler.transform(X_train)
    model.fit(rescaledX, Y_train.ravel())
    rescaledtestX = scaler.transform(X_test)
    predictions = model.predict(rescaledtestX)
    MSE_model = mean_squared_error(Y_test.ravel(), predictions)
    print("MSE is: " + str(MSE_model))
    return

def evaluate_xgb_model(model):
    eval_set = [(X_train, Y_train), (X_test, Y_test)]
    model.fit(X_train, Y_train,
          early_stopping_rounds = 5,
          eval_metric="rmse", 
          eval_set=eval_set, verbose=True)
    Y_pred = model.predict(X_test)
    MSE_model = mean_squared_error(Y_test, Y_pred)
    print("MSE is: " + str(MSE_model))
    return

def Test_Kaggle_Data_XGB(model):
    data_test = pd.read_csv("D:/Python/Datasets/Semiconductor_Data_Test.csv")
    data_test = data_test.drop(columns = ['id'])
    cols_to_transform = ['spacegroup']
    data_test_2 = pd.get_dummies(data_test, columns = cols_to_transform )
    data_test_3 = data_test_2.copy()
    X_Kaggle_test = data_test_3.copy().values
    Predictions = model.predict(X_Kaggle_test)
    Bandgap_Predictions = pd.DataFrame(Predictions)
    return Bandgap_Predictions

def Test_Kaggle_Data(model):
    scaler = StandardScaler().fit(X_train)
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

def KNN_Model(N_Clusters):
    scaler = StandardScaler().fit(X_train)
    rescaledX = scaler.transform(X_train)
    KNN_model = KNeighborsRegressor(n_neighbors = N_Clusters)
    KNN_model.fit(rescaledX, Y_train.ravel())
    rescaledtestX = scaler.transform(X_test)
    KNN_predictions = KNN_model.predict(rescaledtestX)
    print(mean_squared_error(Y_test.ravel(), KNN_predictions))
    return KNN_model


###################################################################

# Execute Code Here

###################################################################


# XGB_Model_12 + KNN-7  - Good Score ! - Improvement from 0.564 to 0.538

xgb_12 = XGBRegressor(n_estimators = 215, max_depth = 4, learning_rate = 0.1,
                     subsample = 0.8,
                     colsample_bytree = 0.5,
                     min_child_weight = 4,
                     reg_alpha = 0.66)

evaluate_xgb_model(xgb_12)

KNN_Model_7 = KNN_Model(7)
evaluate_model(KNN_Model_7)

XGB_Predictions = Test_Kaggle_Data_XGB(xgb_12)
KNN_7_Predictions = Test_Kaggle_Data(KNN_Model_7)
Combined_Predictions_1 = pd.concat((XGB_Predictions, KNN_7_Predictions), axis = 1)
Combined_Predictions_1.columns = ['XGB_Predictions', 'KNN_Predictions']
Combined_Predictions_1["Average"] = (Combined_Predictions_1["XGB_Predictions"] + Combined_Predictions_1["KNN_Predictions"])/2
Combined_Predictions_1.to_csv("Combined Predictions - XGB_12 & KNN_7.csv")


# XGB_Model_12 + KNN-13 - Inferior

xgb_12 = XGBRegressor(n_estimators = 215, max_depth = 4, learning_rate = 0.1,
                     subsample = 0.8,
                     colsample_bytree = 0.5,
                     min_child_weight = 4,
                     reg_alpha = 0.66)

evaluate_xgb_model(xgb_12)

KNN_Model_13 = KNN_Model(13)
evaluate_model(KNN_Model_13)

XGB_Predictions = Test_Kaggle_Data_XGB(xgb_12)
KNN_13_Predictions = Test_Kaggle_Data(KNN_Model_13)
Combined_Predictions_2 = pd.concat((XGB_Predictions, KNN_13_Predictions), axis = 1)
Combined_Predictions_2.columns = ['XGB_Predictions', 'KNN_Predictions']
Combined_Predictions_2["Average"] = (Combined_Predictions_2["XGB_Predictions"] + Combined_Predictions_2["KNN_Predictions"])/2
Combined_Predictions_2.to_csv("Combined Predictions - XGB_12 & KNN_13.csv")


# KNN 7 + SVR_2 - Inferior


KNN_Model_7 = KNN_Model(7)
evaluate_model(KNN_Model_7)

SVR_2 = SVR(C=4.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',
            kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)

evaluate_model(SVR_2)

SVR_Predictions = Test_Kaggle_Data(SVR_2)
KNN_7_Predictions = Test_Kaggle_Data(KNN_Model_7)
Combined_Predictions_3 = pd.concat((SVR_Predictions, KNN_7_Predictions), axis = 1)
Combined_Predictions_3.columns = ['SVR_Predictions', 'KNN_Predictions']
Combined_Predictions_3["Average"] = (Combined_Predictions_3["SVR_Predictions"] + Combined_Predictions_3["KNN_Predictions"])/2
Combined_Predictions_3.to_csv("Combined Predictions - SVR_2 & KNN_7.csv")








