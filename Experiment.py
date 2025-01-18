# from src.Adversial_Multimedia_Recommendation.AMR import AMR
# from src.components.data_transformation import Data_Transformation
# from src.entity.config_entity import DataIngestionConfig
# from src.entity.config_entity import FeatureExtractionConfig
# import numpy as np

# dt = Data_Transformation(DataIngestionConfig.data_store_file_path)
# user_item = dt.User_item()
# item_ids , image = dt.Image_item()
# image_feature = np.load(FeatureExtractionConfig.feature_store_file_path)

# amr = AMR(image_feature,user_item,item_ids)

# from src.components.data_ingestion import DataIngestion
# data = DataIngestion()
# data.initiate_data_ingestion()



# handler = MLflowHandler(
#     tracking_uri=tracking_uri,
#     repo_owner=repo_owner,
#     repo_name=repo_name
# )
# from src.Adversial_Multimedia_Recommendation.AMR_Model import AMRModel
# from src.entity.config_entity import ModelsaveConfig
# model = AMRModel.load_model(ModelsaveConfig.model_store_file_path)
# x = model.predict(None,('AFGDRVPCP742YM5MMLFIKZCGNNRQ',6))
# print(x)


# from dotenv import load_dotenv
# import os
# # Load the .env file
# load_dotenv()
# # Access the tracking URI
# tracking_uri = os.getenv("TRACKING_URI")

# print("Tracking URI:", tracking_uri)