import os
import sys
import pickle
import pandas as pd
import numpy as np
import yaml

from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)


    except Exception as e: 
        raise CustomException(e, sys)   


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV

def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            logging.info(f"model:{model} is started")

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            logging.info(f"model:{model} is Evaluated and best param is {gs.best_params_}")

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            logging.info(f"model:{model} prediction is completed")

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

            logging.info(f"model:{model} score is stored and the test score is : {test_model_score}")
            logging.info(f"model:{model} score is stored and the train score is : {train_model_score}")

        return report

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return pickle.load(file_obj)
        
    except Exception as e: 
        logging.info("Error Occured during load object ")
        raise CustomException(e, sys)
    

# Load YAML configuration
def load_config(config_file_path):
    try:
        with open(config_file_path, 'r') as file:
          return yaml.safe_load(file)
        
    except Exception as e: 
        logging.info("Error Occured during load_config ")
        raise CustomException(e, sys)