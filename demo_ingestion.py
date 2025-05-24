import os
import sys
import mysql.connector
from mysql.connector import Error
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from typing import Tuple
from src.logger import logging
from src.exception import CustomException
from src.utils import load_config   

# Load the configuration
config = load_config('xyz.yaml')

# Data Ingestion Configuration
@dataclass
class DataIngestionConfig:
    raw_data_path: str
    train_data_path: str
    test_data_path: str
    source_data_path: str  # This will now be a database connection string or config

# Data Ingestion Class
class DataIngestion:
    def __init__(self, config):
        self.ingestion_config = DataIngestionConfig(
            raw_data_path=config['data_ingestion']['raw_data_path'],
            train_data_path=config['data_ingestion']['train_data_path'],
            test_data_path=config['data_ingestion']['test_data_path'],
            source_data_path=config['paths']['source_data_path']  # This should now contain DB connection info
        )

    def fetch_data_from_mysql(self):
        """Fetch data from MySQL database"""
        try:
            logging.info("Connecting to MySQL database...")

            # Extracting database configuration from `source_data_path` in config
            host = self.ingestion_config.source_data_path['host']
            database = self.ingestion_config.source_data_path['database']
            user = self.ingestion_config.source_data_path['user']
            password = self.ingestion_config.source_data_path['password']

            # Create a connection to the database
            connection = mysql.connector.connect(
                host=host,
                database=database,
                user=user,
                password=password
            )

            if connection.is_connected():
                logging.info("Successfully connected to the database")

                # Fetch data from a table (example: `data_table`)
                query = "SELECT * FROM data_table"
                df = pd.read_sql(query, connection)

                logging.info(f"Fetched {len(df)} rows from the database")

                return df
        except Error as e:
            logging.error(f"Error connecting to MySQL database: {str(e)}")
            raise CustomException(f"Error connecting to MySQL database: {str(e)}", sys) from e
        finally:
            if connection.is_connected():
                connection.close()
                logging.info("MySQL connection closed")

    def initiate_data_ingestion(self) -> Tuple[str, str]:
        logging.info('Data Ingestion method starts')

        try:
            logging.info("Loading source data from MySQL...")
            # Fetch the data from MySQL
            df = self.fetch_data_from_mysql()

            # Check if directories exist and create if needed
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path), exist_ok=True)

            # Save the raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info(f"Raw data saved at: {self.ingestion_config.raw_data_path}")

            # Perform train-test split
            logging.info("Performing train-test split...")
            train_set, test_set = train_test_split(df, test_size=0.30, random_state=42)

            # Save the train and test datasets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info(f"Train data saved at: {self.ingestion_config.train_data_path}")
            logging.info(f"Test data saved at: {self.ingestion_config.test_data_path}")

            logging.info('Data Ingestion process completed successfully.')

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            logging.error(f"Error occurred during Data Ingestion: {str(e)}")
            raise CustomException(f"Error occurred during Data Ingestion: {str(e)}", sys) from e

if __name__ == "__main__":
    try:
        model_evaluation = DataIngestion(config)
        model_evaluation.initiate_data_ingestion()
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")



