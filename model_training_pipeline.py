'''Pipeline For the Model Training'''

# Importring Required Libraries
import os
import json
import mlflow
import mlflow.pyfunc
import numpy as np
import cornac
import dagshub
from dotenv import load_dotenv
from dataclasses import asdict
from cornac.experiment import Experiment
from src.exception import MyException
from src.logger import logging
from src.entity.config_entity import DataIngestionConfig, FeatureExtractionConfig, ModelsaveConfig, MetricsaveConfig
from src.Adversial_Multimedia_Recommendation.AMRParameters import AMRParameters
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import Data_Transformation
from src.Vission_Transformer_ViT.image_feature_extractor import ImageFeatureExtractor
from src.Adversial_Multimedia_Recommendation.AMR import AMR
from src.Adversial_Multimedia_Recommendation.AMRunner import RunExp
from src.Adversial_Multimedia_Recommendation.AMR_Model import AMRModel

# Load MLflow credentials
load_dotenv()
tracking_uri = os.getenv("TRACKING_URI")
repo_owner = os.getenv("REPO_OWNER")
repo_name = os.getenv("REPO_NAME")

# Setup MLflow
experiment_name = 'Experiment_1'
dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(experiment_name)

# Setup directory for storing best recall
directory = "artifact/metric"
os.makedirs(directory, exist_ok=True)
BEST_RECALL_FILE = os.path.join(directory, "best_recall.json")

# Load previous best recall from file or initialize it
if os.path.exists(BEST_RECALL_FILE):
    with open(BEST_RECALL_FILE, "r") as f:
        best_recall_data = json.load(f)
    previous_best_recall = best_recall_data.get("Recall", 0.0)
else:
    previous_best_recall = 0.0

"""--------Run With Mlflow-------------------"""    

with mlflow.start_run():
    try:
        # Log parameters
        params = AMRParameters()
        params_dict = asdict(params)
        mlflow.log_params(params_dict)

        """_______________Data Ingestion___________________"""
        logging.info("Entered Data Ingestion Pipeline")
        mlflow.log_param("Data_Ingestion", "Started")
        # data = DataIngestion()
        # data.initiate_data_ingestion()
        # mlflow.log_param("Data_Ingestion", "Completed")
        logging.info("Exiting Data Ingestion Pipeline")


        """____________Data Transformation_____________________"""
        logging.info("Entered Data Transformation Pipeline")
        dt = Data_Transformation(DataIngestionConfig.data_store_file_path)
        user_item = dt.User_item()
        item_ids, image = dt.Image_item()
        # Extract the image features by Vission Transformer
        # imfe = ImageFeatureExtractor()
        # imfe.extract_features(image)
        # Loading the image feature
        image_feature = np.load(FeatureExtractionConfig.feature_store_file_path)
        logging.info("Image Features Loaded")
        logging.info("Exiting Data Transformation Pipeline")


        """________________Model Building_________________________"""
        logging.info("Entered into Model Building Component")
        model = AMR(image_feature, user_item, item_ids)
        ratio_split = model.get_ratio_split()
        rec_k = cornac.metrics.Recall(k=AMRParameters.K)
        precis_k = cornac.metrics.Precision(k=AMRParameters.K)
        logging.info("Exiting the Model Building Component")

        """_____________Model Training & Evaluation_____________________"""
        logging.info("Entered into Model Training & Evaluation Component")
        exp = Experiment(eval_method=ratio_split, models=[model.get_model()], metrics=[rec_k, precis_k])
        runner = RunExp(exp, k=AMRParameters.K)
        metrics = runner.run_experiment()

        # Log metrics
        prec = metrics[f"Precision@{AMRParameters.K}"]
        rec = metrics[f"Recall@{AMRParameters.K}"]
        mlflow.log_metric("Precision", prec)
        mlflow.log_metric("Recall", rec)

        """__________________________Storing model component____________________________"""
        # Update the best recall score if the current one is better
        if rec > previous_best_recall:
            logging.info(f"New best recall achieved: {rec} (previous: {previous_best_recall})")
            previous_best_recall = rec

            # Save the updated best recall to the file
            with open(BEST_RECALL_FILE, "w") as f:
                json.dump({"Precision": prec, "Recall": previous_best_recall}, f)

            # Save and log the model
            logging.info("Saving the model as it achieved a new best recall score.")
            amr = model.get_model()
            amr_model = AMRModel(amr)
            amr_model.save_model(ModelsaveConfig.model_store_file_path)
            mlflow.pyfunc.log_model("AMR_Model", python_model=amr_model)
        else:
            logging.info(f"Recall did not improve. Current: {rec}, Best: {previous_best_recall}")

        logging.info("Exited Model Training & Evaluation Component")
        mlflow.end_run()

    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        mlflow.log_param("Pipeline_Status", "Failed")
        raise MyException(e)
