import os
import sys
import pickle
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

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


def evaluate_model(X_train, y_train, X_test, y_test, models, param):
    """
    Evaluates multiple models using GridSearchCV for hyperparameter tuning.
    Logs the results and model artifacts to MLflow.

    Args:
        X_train (array): Training feature data.
        y_train (array): Training target data.
        X_test (array): Testing feature data.
        y_test (array): Testing target data.
        models (dict): Dictionary of model names and corresponding model objects.
        param (dict): Dictionary of model names and their respective hyperparameter grids.

    Returns:
        dict: A dictionary containing model names as keys and their R2 scores as values.
    """
    try:
        report = {}
        
        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            para = param.get(model_name, {})

            logging.info(f"Evaluating model: {model_name}")

            # Start an MLflow run for the current model
            with mlflow.start_run(run_name=model_name):
                # Perform GridSearchCV for hyperparameter tuning
                gs = GridSearchCV(model, para, cv=3, n_jobs=-1, scoring='r2', verbose=1)
                gs.fit(X_train, y_train)

                # Set the best parameters to the model
                model.set_params(**gs.best_params_)
                model.fit(X_train, y_train)

                logging.info(f"Model: {model_name} best parameters: {gs.best_params_}")

                # Log best parameters to MLflow
                mlflow.log_params(gs.best_params_)

                # Make predictions on the test set
                y_pred = model.predict(X_test)
                logging.info(f"Model: {model_name} predictions completed")

                # Calculate R2 score
                test_model_score = r2_score(y_test, y_pred)
                report[model_name] = test_model_score

                logging.info(f"Model: {model_name} R2 score: {test_model_score}")

                # Log the R2 score to MLflow
                mlflow.log_metric("R2_score", test_model_score)
                mlflow.log_metric("MSE", mean_squared_error(y_test,y_pred))               

                # Log the trained model to MLflow
                mlflow.sklearn.log_model(model, model_name)

        return report  

    except Exception as e: 
        logging.error("Error occurred during model evaluation")
        raise CustomException(e, sys)
    

def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return pickle.load(file_obj)
        
    except Exception as e: 
        logging.info("Error Occured during load object ")
        raise CustomException(e, sys)