import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from src.logger import logging
from src.exception import CustomException
from src.utils import save_obj, load_config

@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path: str
    train_array_file_path: str
    test_array_file_path: str

class DataTransformation:
    def __init__(self, config):
        self.data_transformation_config = DataTransformationConfig(
            preprocessor_ob_file_path=config['data_transformation']['preprocessor_ob_file_path'],
            train_array_file_path=config['data_transformation']['train_array_file_path'],
            test_array_file_path=config['data_transformation']['test_array_file_path']
        )

        # Ensure directories for all paths exist
        for path in [self.data_transformation_config.train_array_file_path,
                     self.data_transformation_config.test_array_file_path]:
            directory = os.path.dirname(path)
            if not os.path.exists(directory):
                try:
                    os.makedirs(directory, exist_ok=True)
                    logging.info(f"Directory created: {directory}")
                except Exception as e:
                    logging.error(f"Error creating directory {directory}: {str(e)}")
                    raise CustomException(e, sys)

    def get_data_transformation_object(self):
        try:
            logging.info("Data Transformation Initiated")
            numerical_columns = ['Ram', 'Weight', 'TouchScreen', 'IPS', 'PPI', 'HDD', 'SSD']
            categorical_columns = ['Company', 'TypeName', 'Cpu_Brand', 'Gpu_brand', 'os']

            num_pipeline = Pipeline(
                steps=[('imputer', SimpleImputer(strategy='median')),
                       ('scalar', StandardScaler(with_mean=False))]
            )

            cat_pipeline = Pipeline(
                steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                       ('onehot', OneHotEncoder(drop="first")),
                       ('scalar', StandardScaler(with_mean=False))]
            )

            processor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns)
            ])

            logging.info("Pipeline completed")
            return processor

        except Exception as e:
            logging.error(f"Error in Pipeline: {str(e)}")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info(f"Reading train data from: {train_path}")
            logging.info(f"Reading test data from: {test_path}")
            
            # Check if train and test files exist before reading
            if not os.path.exists(train_path):
                raise FileNotFoundError(f"Train file not found: {train_path}")
            if not os.path.exists(test_path):
                raise FileNotFoundError(f"Test file not found: {test_path}")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading of train and test data has been completed")
            logging.info(f"Train DataFrame Head: \n{train_df.head().to_string()}")
            logging.info(f"Test DataFrame Head: \n{test_df.head().to_string()}")

            logging.info("Obtaining Preprocessor Object")
            preprocessor_Obj = self.get_data_transformation_object()

            target_column = 'Price'
            input_feature_train_df = train_df.drop(columns=[target_column])
            target_feature_train_df = train_df[target_column]
            input_feature_test_df = test_df.drop(columns=[target_column])
            target_feature_test_df = test_df[target_column]

            logging.info("Applying Preprocessor object to the train and test datasets")
            input_feature_train_arr = preprocessor_Obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_Obj.transform(input_feature_test_df)

            # Combine features and target arrays
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Saving arrays to .npy files
            logging.info(f"Saving train array to: {self.data_transformation_config.train_array_file_path}")
            np.save(self.data_transformation_config.train_array_file_path, train_arr)
            logging.info(f"Train array successfully saved.")

            logging.info(f"Saving test array to: {self.data_transformation_config.test_array_file_path}")
            np.save(self.data_transformation_config.test_array_file_path, test_arr)
            logging.info(f"Test array successfully saved.")

            # Saving the preprocessor object
            save_obj(
                file_path=self.data_transformation_config.preprocessor_ob_file_path,
                obj=preprocessor_Obj
            )
            logging.info(f"Preprocessor object saved at: {self.data_transformation_config.preprocessor_ob_file_path}")

            # Check if train and test files still exist after processing
            if not os.path.exists(train_path):
                logging.warning(f"Train file was deleted during processing: {train_path}")
            if not os.path.exists(test_path):
                logging.warning(f"Test file was deleted during processing: {test_path}")

            return train_arr, test_arr, self.data_transformation_config.preprocessor_ob_file_path

        except Exception as e:
            logging.error(f"Error occurred in initiating data transformation: {str(e)}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        config = load_config('xyz.yaml')
        data_transformation = DataTransformation(config)
        train_path = config['data_ingestion']['train_data_path']
        test_path = config['data_ingestion']['test_data_path']

        train_arr, test_arr, preprocessor_ob_path = data_transformation.initiate_data_transformation(train_path, test_path)

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
