import os
import sys
import pickle
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging

from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)


    except Exception as e: 
        raise CustomException(e, sys)   


def evaluate_model(X_train,y_train, X_test, y_test, models):
    try:
        report = {}
        for i in range(len(list(models))):
             model = list(models.values())[i]
             model.fit(X_test, y_test)

            # Make Prediction

             y_pred = model.predict(X_test)

             test_model_score = r2_score(y_test,y_pred)
             report[list(models.keys())[i]] = test_model_score

        return report  

    except Exception as e: 
        logging.info("Error Occured during model Training ")
        raise CustomException(e, sys)
    

def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return pickle.load(file_obj)
        
    except Exception as e: 
        logging.info("Error Occured during load object ")
        raise CustomException(e, sys)