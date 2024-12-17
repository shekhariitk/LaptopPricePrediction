import os
import sys
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging


from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainerClass

if __name__ == '__main__':
    obj = DataIngestion()
    # Call the initiate_data_ingestion method
    result = obj.initiate_data_ingestion()

# Check if the result is None
    if result is None:
       logging.error("Data Ingestion failed. Please check the logs for more details.")
    # Handle the failure, possibly exit or raise an exception
    else:
        train_data_path, test_data_path = result
        logging.info(f"Data Ingestion Successful: Train path - {train_data_path}, Test path - {test_data_path}")


    data_transformation = DataTransformation()
    train_arr ,test_arr,_ = data_transformation.initiate_data_transformation(train_data_path,test_data_path)

    model_trainer  = ModelTrainerClass()
    model_trainer.initiate_model_training(train_arr,test_arr)
