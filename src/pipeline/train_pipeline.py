from src.components.data_ingestion import DataIngestion
from src.exception import CustomException


class TrainPipeline:
    def start(self):
        try:
            data_ingestion = DataIngestion()
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

            from src.components.data_transformation import DataTransformation

            data_transformation = DataTransformation()
            train_arr, test_arr, _, = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

            from src.components.model_trainer import ModelTrainer

            model_trainer = ModelTrainer()
            return model_trainer.initiate_model_trainer(train_arr, test_arr)
        except Exception as e:
            raise CustomException(e, e)

