from src.student_score_mlflow.logger import logging
from src.student_score_mlflow.exception import CustomException
from src.student_score_mlflow.components.data_ingestion import DataIngestion
from src.student_score_mlflow.components.data_ingestion import DataIngestionConfig
from src.student_score_mlflow.components.data_transformation import DataTransformationConfig
from src.student_score_mlflow.components.data_transformation import DataTransformation
from src.student_score_mlflow.components.model_trainer import ModelTrainerConfig
from src.student_score_mlflow.components.model_trainer import ModelTrainer
import sys

if __name__=="__main__":
    logging.info("The execution has started")

    try:
        #data_ingestion_config=DataIngestionConfig()
        data_ingestion=DataIngestion()
        train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()

        data_transformation_config=DataTransformationConfig()
        data_transformation=DataTransformation()
        train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data_path,test_data_path)

        # model training
        model_trainer=ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_arr,test_arr)) # print best model r2_score

    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)
