# from src.Adversial_Multimedia_Recommendation.AMR import AMR
# from src.components.data_transformation import Data_Transformation
# from src.entity.config_entity import DataIngestionConfig
# from src.entity.config_entity import FeatureExtractionConfig
import numpy as np

# dt = Data_Transformation(DataIngestionConfig.data_store_file_path)
# user_item = dt.User_item()
# item_ids , image = dt.Image_item()
# image_feature = np.load(FeatureExtractionConfig.feature_store_file_path)

# amr = AMR(image_feature,user_item,item_ids)

# from src.components.data_ingestion import DataIngestion
# data = DataIngestion()
# data.initiate_data_ingestion()

# from src.utils.mlflow_handler import MLflowHandler
# tracking_uri = "https://dagshub.com/AnkurK07/Adversarial-Multimedia-Rec-Visual.mlflow"
# experiment_name = 'Experiment_1'
# repo_owner='AnkurK07'
# repo_name='Adversarial-Multimedia-Rec-Visual'

# handler = MLflowHandler(
#     tracking_uri=tracking_uri,
#     repo_owner=repo_owner,
#     repo_name=repo_name
# )
from src.Adversial_Multimedia_Recommendation.AMR_Model import AMRModel
from src.entity.config_entity import ModelsaveConfig
model = AMRModel.load_model(ModelsaveConfig.model_store_file_path)
x = model.predict(None,('AFGDRVPCP742YM5MMLFIKZCGNNRQ',6))
print(x)