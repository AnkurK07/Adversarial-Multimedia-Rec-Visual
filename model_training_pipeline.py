'''Pipeline for the model training'''

import mlflow
import mlflow.pyfunc
import numpy as np
import cornac
import dagshub
import json
import os

from dataclasses import asdict
from cornac.experiment import Experiment
from src.exception import MyException
from src.logger import logging

from src.entity.config_entity import DataIngestionConfig
from src.entity.config_entity import FeatureExtractionConfig
from src.entity.config_entity import ModelsaveConfig
from src.entity.config_entity import MetricsaveConfig
from src.Adversial_Multimedia_Recommendation.AMRParameters import AMRParameters

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import Data_Transformation
from src.Vission_Transformer_ViT.image_feature_extractor import ImageFeatureExtractor
from src.Adversial_Multimedia_Recommendation.AMR import AMR
from src.Adversial_Multimedia_Recommendation.AMRunner import RunExp
from src.Adversial_Multimedia_Recommendation.AMR_Model import AMRModel

# Getting mlflow crediantials
tracking_uri = os.getenv("tracking_uri")
repo_owner = os.getenv("repo_owner")
repo_name = os.getenv("repo_name")

# setting up the mlflow
experiment_name ='Experiment_1'
dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
mlflow.set_tracking_uri(tracking_uri)

# Initialize MLflow
mlflow.set_experiment(experiment_name)
with mlflow.start_run():
    try:
        '''Tracking parameters'''
        params = AMRParameters()
        params_dict = asdict(params)
        mlflow.log_params(params_dict)

        '''Data Ingestion Component'''
        logging.info("Entered Data Ingestion Pipeline")
        mlflow.log_param("Data_Ingestion", "Started")
        # data = DataIngestion()
        # data.initiate_data_ingestion()
        #mlflow.log_param("Data_Ingestion", "Completed")
        logging.info("Exiting Data Ingestion Pipeline")

        '''Data Transformation Component'''

        logging.info("Entered into Data Transformation Pipeline")
        # Make data into the required formet
        dt = Data_Transformation(DataIngestionConfig.data_store_file_path)
        user_item = dt.User_item()
        item_ids , image = dt.Image_item()
        # Extract the image features by Vission Transformer
        # imfe = ImageFeatureExtractor()
        # imfe.extract_features(image)
        # Loading the image feature
        image_feature = np.load(FeatureExtractionConfig.feature_store_file_path)
        logging.info("Image Features Loaded")
        logging.info("Exiting Data Transformation Pipeline")

        '''Model Building Component'''
        logging.info("Entered into Model Building Component")
        # Initializing the model
        model = AMR(image_feature,user_item,item_ids)
        # Getting ratio split
        ratio_split = model.get_ratio_split()
        # Defining the metric parameters
        rec_k = cornac.metrics.Recall(k=AMRParameters.K)
        precis_k = cornac.metrics.Precision(k=AMRParameters.K)
        logging.info("Exiting the Model Building Component")

        '''Model Training & Evaluation Component'''
        logging.info("Entered into Model Training  & Evaluation Component ]")
        # Put everything together into an experiment
        exp = Experiment(eval_method=ratio_split,models=[model.get_model()], metrics=[rec_k,precis_k])
        # Define the runner
        runner = RunExp(exp, k=AMRParameters.K)
        # Start training and store metrics
        metrics = runner.run_experiment()
        # Tracking Matrices
        prec = metrics[f"Precision@{AMRParameters.K}"]
        rec  = metrics[f"Recall@{AMRParameters.K}"]
        mlflow.log_metric("Precision",prec)
        mlflow.log_metric("Recall",rec)
        # Save the matrices
        runner.save_metrics(MetricsaveConfig.metric_store_file_path)
        logging.info("Exited Model Training & Evaluation Component")

        '''Model Saving Component'''
        logging.info("Entered into Model Saving Component")
        # Getting Model
        amr = model.get_model()
        # Making Compatible with mlflow
        amr_model = AMRModel(amr)
        # Saving Model
        logging.info("Model is being saved......")
        amr_model.save_model(ModelsaveConfig.model_store_file_path)
        logging.info("Model Saved Successfully")
        # Logging Model
        logging.info("Model mlflow logging started.........")
        mlflow.pyfunc.log_model("AMR_Model", python_model=amr_model)
        logging.info("Model logged successfully")
        logging.info("Exited Model Saving Component")
        mlflow.end_run()

    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        mlflow.log_param("Pipeline_Status", "Failed")
        raise MyException(e)
