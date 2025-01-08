import yaml
import sys
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from src.logger import logging
from src.exception import CustomException
from src.utils import save_obj

## Load YAML configuration
def load_config(config_file_path):
    with open(config_file_path, 'r') as file:
        return yaml.safe_load(file)

# Load the configuration
config = load_config('xyz.yaml')

# Data Transformation Configuration
@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path: str = config['data_transformation']['preprocessor_ob_file_path']

## Data Transformation Class
class DataTransformation:
    def __init__(self, config):
        self.data_transformation_config = DataTransformationConfig(
            preprocessor_ob_file_path=config['data_transformation']['preprocessor_ob_file_path']
        )

    def get_data_transformation_object(self):
        try:
            logging.info("Data Transformation Initiated")
            numerical_columns = ['Ram', 'Weight', 'TouchScreen', 'IPS', 'PPI', 'HDD', 'SSD']
            categorical_columns = ['Company', 'TypeName', 'Cpu_Brand', 'Gpu_brand', 'os']

            # Numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scalar', StandardScaler(with_mean=False))
                ]
            )

            # Categorical pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ordinal', OneHotEncoder(sparse=False, drop="first")),
                    ('scalar', StandardScaler(with_mean=False))
                ]
            )

            processor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns)
            ])         
            logging.info("Pipeline completed")
            return processor

        except Exception as e:
            logging.error("Error in Pipeline")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            ## Reading the train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading of train and test data has been completed")
            logging.info(f"Train DataFrame Head: \n {train_df.head().to_string()}")
            logging.info(f"Test DataFrame Head: \n {test_df.head().to_string()}")

            logging.info("Obtaining Preprocessor Object")

            preprocessor_Obj = self.get_data_transformation_object()

            target_column = 'Price'

            input_feature_train_df = train_df
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df
            target_feature_test_df = test_df[target_column]

            ## Applying Transformation
            input_feature_train_arr = preprocessor_Obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_Obj.transform(input_feature_test_df)

            logging.info("Applying Preprocessor object to the train and test datasets")

            ## Concatenating the train and test array
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_obj(
                file_path=self.data_transformation_config.preprocessor_ob_file_path,
                obj=preprocessor_Obj
            )

            logging.info("Preprocessor is created and saved")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_file_path
            )

        except Exception as e:
            logging.error("Error occurred in initiating data transformation")
            raise CustomException(e, sys)
