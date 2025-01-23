# defining the configuration

import os
from src.constants import *
from dataclasses import dataclass
from datetime import datetime

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = os.path.join(ARTIFACT_DIR)
    timestamp: str = TIMESTAMP


training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()

# Data Ingestion data class
@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME)
    data_store_file_path: str = os.path.join(data_ingestion_dir, FILE_NAME)
    collection_name:str = DATA_INGESTION_COLLECTION_NAME

# Feature extraction data class
@dataclass
class FeatureExtractionConfig:
    feature_extraction_dir: str = os.path.join(training_pipeline_config.artifact_dir, FEATURE_EXTRACTION_DIR_NAME)
    feature_store_file_path: str =  os.path.join(feature_extraction_dir,FEATURE_FILE_NAME)

# Model saving data class   
class ModelsaveConfig:
    model_saving_dir: str = os.path.join(training_pipeline_config.artifact_dir, MODEL_SAVING_DIR_NAME)
    model_store_file_path: str =  os.path.join(model_saving_dir,MODEL_FILE_NAME)

# Metric saving data class
class MetricsaveConfig:
     metric_saving_dir: str = os.path.join(training_pipeline_config.artifact_dir, METRIC_SAVING_DIR_NAME)
     metric_store_file_path: str =  os.path.join(metric_saving_dir)


    
# Data Class for model pusher
@dataclass
class ModelPusherConfig:
    bucket_name: str = MODEL_BUCKET_NAME
    s3_model_key_path: str = MODEL_FILE_NAME

# Data Class For Prediction
@dataclass
class VehiclePredictorConfig:
    model_file_path: str = MODEL_FILE_NAME
    model_bucket_name: str = MODEL_BUCKET_NAME