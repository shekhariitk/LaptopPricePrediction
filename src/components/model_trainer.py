import os
import sys
import yaml
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj, evaluate_models
from dataclasses import dataclass
from jsonschema import validate, ValidationError

# Model Training Configuration
@dataclass
class ModelTrainerconfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    param_file_path = os.path.join("config", "param.yaml")
    schema_file_path = os.path.join("config", "schema.yaml")

# Model Training Class
class ModelTrainerClass:
    def __init__(self):
        self.model_trainer_config = ModelTrainerconfig()

    def load_params(self):
        """Load hyperparameters from the param.yaml file."""
        try:
            with open(self.model_trainer_config.param_file_path, 'r') as file:
                params = yaml.safe_load(file)
            return params['models']
        except Exception as e:
            logging.error(f"Error while loading parameters from {self.model_trainer_config.param_file_path}")
            raise CustomException(e, sys)

    def validate_params(self, params):
        """Validate parameters against schema.yaml."""
        try:
            with open(self.model_trainer_config.schema_file_path, 'r') as schema_file:
                schema = yaml.safe_load(schema_file)

            validate(instance=params, schema=schema)
            logging.info("Parameter validation successful.")
        except ValidationError as ve:
            logging.error(f"Parameter validation failed: {ve.message}")
            raise CustomException(f"Parameter validation failed: {ve.message}", sys)
        except Exception as e:
            logging.error(f"Error during schema validation: {str(e)}")
            raise CustomException(e, sys)

    def initiate_model_training(self, train_arr, test_arr):
        try:
            logging.info("Splitting independent and dependent variables.")
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                np.log(train_arr[:, -1]),
                test_arr[:, :-1],
                np.log(test_arr[:, -1])
            )

            models = {
                "LinearRegression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "ElasticNet": ElasticNet(),
                "RandomForestRegressor": RandomForestRegressor(),
                "KNeighborsRegressor": KNeighborsRegressor(),
                "GradientBoostingRegressor": GradientBoostingRegressor(),
                "AdaBoostRegressor": AdaBoostRegressor(),
                "ExtraTreesRegressor": ExtraTreesRegressor(),
                "SVR": SVR(),
                "XGBRegressor": XGBRegressor(n_estimators=45, max_depth=5, learning_rate=0.5)
            }

            # Load and validate parameters
            params = self.load_params()
            self.validate_params(params)  # Validate params from param.yaml

            # Evaluate models
            model_report = evaluate_models(X_train, y_train, X_test, y_test, models, param=params)
            logging.info(f"Model report: {model_report}")

            # Get best model from the report
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            logging.info(f"Best model found: {best_model_name} with R2 Score: {best_model_score}")

            # Save the best model
            save_obj(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)

        except Exception as e:
            logging.error("Error occurred during model training.")
            raise CustomException(e, sys)


