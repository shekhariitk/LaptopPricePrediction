import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from typing import Tuple

## Initialize the data ingestion configuration
@dataclass
class DataIngestionconfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')


## Create a data ingestion class
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionconfig()

    def initiate_data_ingestion(self) -> Tuple[str, str]:
        logging.info('Data Ingestion method starts')

        try:
            # Reading the dataset
            df = pd.read_csv(os.path.join('notebooks', 'data', 'cleaned_data.csv'))
            logging.info('Dataset read as pandas DataFrame')

            # Create the required directories if they don't exist
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # Saving the raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Raw data saved at: %s", self.ingestion_config.raw_data_path)

            # Perform train-test split
            logging.info("Train-test split in progress")
            train_set, test_set = train_test_split(df, test_size=0.30, random_state=42)

            # Save the train and test datasets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Data Ingestion is completed successfully')

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            logging.error(f"Error occurred during Data Ingestion: {str(e)}")
            raise CustomException(f"Error occurred during Data Ingestion: {str(e)}", sys) from e