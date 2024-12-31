import os
import sys
import pickle
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub

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
        best_model_name = None
        best_model_score = float("-inf")
        best_model = None

        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            para = param.get(model_name, {})

            logging.info(f"Evaluating model: {model_name}")

            try:
                # Start an MLflow run for the current model
                mlflow.set_tracking_uri("https://dagshub.com/shekhariitk/LaptopPricePrediction.mlflow")
                mlflow.set_experiment("Model_Evaluation")
                dagshub.init(repo_owner='shekhariitk', repo_name='LaptopPricePrediction', mlflow=True)
                with mlflow.start_run(run_name=model_name, nested=True):
                    if not para:
                        logging.info(f"No hyperparameters provided for {model_name}. Training without GridSearchCV.")
                        model.fit(X_train, y_train)
                    else:
                        # Perform GridSearchCV for hyperparameter tuning
                        gs = GridSearchCV(model, para, cv=3, n_jobs=-1, scoring='r2', verbose=0)
                        gs.fit(X_train, y_train)
                        model.set_params(**gs.best_params_)

                    # Train the model on the training data
                    model.fit(X_train, y_train)

                    # Log best parameters to MLflow
                    if para:
                        mlflow.log_params(gs.best_params_)
                        logging.info(f"Model: {model_name} best parameters: {gs.best_params_}")

                    # Make predictions on the test set
                    y_pred = model.predict(X_test)
                    logging.info(f"Model: {model_name} predictions completed")

                    # Calculate R2 score and other metrics
                    r2 = r2_score(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)

                    report[model_name] = r2

                    logging.info(f"Model: {model_name} R2 score: {r2}")

                    # Log metrics to MLflow
                    mlflow.log_metric("R2_score", r2)
                    mlflow.log_metric("MSE", mse)
                    mlflow.log_metric("MAE", mae)

                    # Log the trained model to MLflow
                    mlflow.sklearn.log_model(model, model_name)

                    # Update the best model if applicable
                    if r2 > best_model_score:
                        best_model_name = model_name
                        best_model_score = r2
                        best_model = model

            except Exception as model_error:
                logging.error(f"Error evaluating model: {model_name}, Error: {str(model_error)}")
                continue

        logging.info(f"Best model: {best_model_name} with R2 Score: {best_model_score}")
        return report, best_model_name, best_model_score, best_model

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