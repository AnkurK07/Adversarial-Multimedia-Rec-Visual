import pandas as pd
from src.logger import logging
from src.exception import MyException

class Data_Transformation:
    def __init__(self, data_store_file_path):
        self.data_store_file_path = data_store_file_path
        
        try:
            # Read the file into a pandas DataFrame
            self.df = pd.read_csv(self.data_store_file_path)
            logging.info(f"Data loaded successfully from {self.data_store_file_path}")
        except Exception as e:
            logging.error(f"Error while loading data from {self.data_store_file_path}: {str(e)}")
            raise MyException(f"Error while loading data from {self.data_store_file_path}: {str(e)}")

    def User_item(self):
        try:
            # Initialize an empty list to store user-item data
            user_item = []
            for index, row in self.df.iterrows():
                user_id = row['user_id']
                item_id = row['item_id']
                rating = row['rating']
                user_item.append((user_id, item_id, rating))
            logging.info("User-item data processed successfully.")
            return user_item
        except Exception as e:
            logging.error(f"Error while processing user-item data: {str(e)}")
            raise MyException(f"Error while processing user-item data: {str(e)}")

    def Image_item(self):
        try:
            # Drop duplicates and extract item_id and image
            df1 = self.df.drop_duplicates(subset=['item_id', 'image'])[['item_id', 'image']]
            item_ids = df1['item_id'].tolist()
            images = df1['image'].tolist()
            logging.info("Unique item_id and image_url extracted successfully.")
            return item_ids, images
        except Exception as e:
            logging.error(f"Error while extracting unique combinations: {str(e)}")
            raise MyException(f"Error while extracting unique combinations: {str(e)}")
