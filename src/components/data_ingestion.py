import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd


from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Use project-root-relative path (robust to different working directories)
            dataset_path = os.path.join(os.getcwd(), 'notebook', 'data', 'stud.csv')
            df = pd.read_csv(dataset_path)

            logging.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            # Avoid sklearn dependency at this stage; do a deterministic split.
            df_shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
            split_idx = int(len(df_shuffled) * 0.8)
            train_set = df_shuffled.iloc[:split_idx]
            test_set = df_shuffled.iloc[split_idx:]

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)


            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e, sys)
    
if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
    