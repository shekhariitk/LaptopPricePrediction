from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler # Handling categorical data
from sklearn.impute import SimpleImputer # handling missing values

import sys, os
import pandas as pd 
import numpy as np

from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException

from src.utils import save_obj

## Data Transformation Configuration
@dataclass

class DataTransformationConfig:
    preprocessor_ob_file_path = os.path.join("artifacts","preprocessor.pkl")



## Data Transformation Class


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()



    def get_data_transformation_object(self):
        try:
            logging.info("Data Transformation Initiated")
            numerical_columns = ['Ram', 'Weight', 'TouchScreen', 'IPS', 'PPI', 'HDD', 'SSD']
            categorical_columns = ['Company', 'TypeName', 'Cpu_Brand', 'Gpu_brand', 'os']

            # Numerical pipeline

            num_pipeline = Pipeline(
                    steps=[
                         ('imputer',SimpleImputer(strategy='median')),
                         ('scalar',StandardScaler(with_mean=False))
                         ]
                         )
# Categorical pipeline

            cat_pipeline = Pipeline(
                    steps=[
                         ('imputer',SimpleImputer(strategy='most_frequent')),
                         ('ordinal',OneHotEncoder(sparse=False,drop="first")),
                         ('scalar',StandardScaler(with_mean=False))
                         ]
                         )

            proccessor = ColumnTransformer([
                         ('num_pipeline',num_pipeline,numerical_columns),
                         ('cat_pipeline', cat_pipeline, categorical_columns)
                         ])         
            return proccessor
            logging.info("Pipeline completed")

        except Exception as e:
            logging.info("Error in Pipeline")
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path, test_path):
        try:
            ## Reading the train and test data
            train_df  = pd.read_csv(train_path)
            test_df   = pd.read_csv(test_path)


            logging.info("Reading of train and test data has been completed")
            logging.info(f"Train DataFrame Head: \n {train_df.head().to_string()}")
            logging.info(f"Test DataFrame Head: \n {test_df.head().to_string()}")



            logging.info("Obtaining Preprocessor Object")

            preprocessor_Obj = self.get_data_transformation_object()

            target_column  =  'Price'

            input_feature_train_df = train_df
            target_feature_train_df = train_df[target_column]



            input_feature_test_df = test_df
            target_feature_test_df = test_df[target_column]


            ## Applying Transformation

            input_feature_train_arr = preprocessor_Obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_Obj.transform(input_feature_test_df)



            logging.info("Applying Preprocessor object to the train and test datasets")


            ## Concatenating the train and test array

            train_arr  = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr  = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]


            save_obj(
                file_path=self.data_transformation_config.preprocessor_ob_file_path,
                obj=preprocessor_Obj
            )

            logging.info("Preprocessor is created and saved")


            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_file_path
            )
        
        except Exception as e:
            logging.info("Error occured in initiating data transformation")
            raise CustomException(e, sys)
        

     
