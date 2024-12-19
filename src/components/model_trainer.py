import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging

# Model Training
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor,ExtraTreesRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor


from src.utils import save_obj
from src.utils import evaluate_model

import os, sys
from dataclasses import dataclass

## Model Trainning Configuration

@dataclass
class ModelTrainerconfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

## Model Training Class
class ModelTrainerClass:
    def __init__(self):
        self.model_trainer_config  = ModelTrainerconfig()

    def  initiate_model_training(self,train_arr, test_arr):
        try:
            logging.info("Splitting independent and Dependent Variable")
            X_train,y_train, X_test, y_test = (
                train_arr[:,:-1],
                np.log(train_arr[:,-1]),
                test_arr[:,:-1],
                np.log(test_arr[:,-1])

            )

            models  = {

                "LinearRegression":LinearRegression(),
                "Lasso":Lasso(),
                "Ridge": Ridge(),
                "DecisionTreeRegressor":DecisionTreeRegressor(),
                "ElasticNet": ElasticNet(),
                "RandomForestRegressor":RandomForestRegressor(),
                "KNeighborsRegressor":KNeighborsRegressor(),
                "DecisionTreeRegressor":DecisionTreeRegressor(),
                "GradientBoostingRegressor":GradientBoostingRegressor(),
                "AdaBoostRegressor":AdaBoostRegressor(),
                "ExtraTreesRegressor":ExtraTreesRegressor(),
                "SVR":SVR(),
                "XGBRegressor":XGBRegressor(n_estimators=45,max_depth=5,learning_rate=0.5)
    
    }
            params={
                "DecisionTreeRegressor": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter':['best','random'],
                    'max_features':['sqrt','log2'],
                },
                "GradientBoostingRegressor":{
                    #'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    #'criterion':['squared_error', 'friedman_mse'],
                    #'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,45,64,128,256]
                },
                "LinearRegression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "AdaBoostRegressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "KNeighborsRegressor":{
                    'n_neighbors': [3, 5, 10, 20],
                    'weights': ['uniform', 'distance'],
                    'metric': ['minkowski', 'euclidean', 'manhattan']
                },
                "ExtraTreesRegressor":{
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Lasso":{
                    'alpha': [np.random.uniform(0.01, 10)], 
                    'max_iter': [1000, 2000, 5000],
                    'fit_intercept': [True, False]},

                "Ridge":{
                    'alpha': [0.1, 1, 10, 100],
                    'solver': ['auto', 'svd', 'lsqr']
                },
                "SVR":{
                    'kernel':['rbf'],
                    'C':[1,10,100,10000],
                    'epsilon':[0.1,0.2]
                },
                "ElasticNet":{},
                "DecisionTreeRegressor":{
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['auto', 'sqrt', 'log2']},

                "RandomForestRegressor":{
                    'n_estimators': [np.random.randint(100, 1000)],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [np.random.randint(2, 20)],
                    'min_samples_leaf': [np.random.randint(1, 20)],
                    'max_features': ['auto', 'sqrt', 'log2'],
                    'bootstrap': [True, False]
                }
                        
                        }
            model_report : dict = evaluate_model(X_train,y_train, X_test, y_test, models,param=params)
            print(model_report)
            print("\n ==================================================================================")
            logging.info(f"Model report info : {model_report}")

            ## To get best model from model dictionary
            best_model_score = max(sorted(model_report.values()))

            best_model_name  = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            print(f"Best model found , Best model name is {best_model_name} and that R2 Score: {best_model_score}")
            print("\n=================================================================")
            logging.info(f"Best model found , Best model name is {best_model_name} and that R2 Score: {best_model_score}")


            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

        except   Exception as e:
            logging.info("Error occured in model trainer path") 
            raise CustomException(e, sys)
        
        