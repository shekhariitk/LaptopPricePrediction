import os
import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainerClass
from src.utils import load_config

# Load the configuration
config = load_config('xyz.yaml')
logging.info(f"Loaded config: {config}")

def main():
    try:
        # Data Ingestion
        logging.info("Starting Data Ingestion...")
        data_ingestion = DataIngestion(config)
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        logging.info(f"Data Ingestion completed. Train data path: {train_data_path}, Test data path: {test_data_path}")

        # Data Transformation
        logging.info("Starting Data Transformation...")
        data_transformation = DataTransformation(config)
        train_arr, test_arr, preprocessor_ob_path = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
        logging.info("Data Transformation completed.")

        # Model Training
        logging.info("Starting Model Training...")
        model_trainer = ModelTrainerClass()
        model_trainer.initiate_model_training(train_arr, test_arr)
        logging.info("Model Training completed.")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()