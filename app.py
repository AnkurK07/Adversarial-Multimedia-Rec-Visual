'''-------------------------------------------Setting Up Streaamlit App----------------------------------------------------------------'''
import streamlit as st
import base64
# Set the background image using the external URL
page_bg_img = f"""
<style>
.stApp {{
    background-image: url("https://img.freepik.com/free-vector/interior-design-dressing-room_1308-55282.jpg?t=st=1737143988~exp=1737147588~hmac=17b04a6d67abe255a5054a6b7eb4f8c6660efed7eb0ad383ebdfe7e7448c384c&w=900");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}
</style>
"""

# Apply the background style
st.markdown(page_bg_img, unsafe_allow_html=True)

# Add content
#st.title("Welcome to Streamlit!")

#-------------------------------------------------------Data for App----------------------------------------------------------------
import pandas as pd
import pymongo
from dotenv import load_dotenv
import os
load_dotenv()
DB_NAME = os.getenv('DB_NAME')
COLLECTION_NAME = os.getenv('COLLECTION_NAME')
CONNECTION_URL = os.getenv('MONGODB_URL')
client = pymongo.MongoClient(CONNECTION_URL)
data_base = client[DB_NAME]
collection = data_base[COLLECTION_NAME]
df1 = pd.DataFrame(list(collection.find()))
df = df1.drop_duplicates(subset=['item_id','image'])[['item_id', 'image']]
user_ids = df1['user_id'].unique().tolist()
#----------------------------------------------------Loading Model For App--------------------------------------------------------------------
import mlflow
import mlflow.pyfunc
import dagshub
from dotenv import load_dotenv
load_dotenv()
tracking_uri = os.getenv("TRACKING_URI")
repo_owner = os.getenv("REPO_OWNER")
repo_name = os.getenv("REPO_NAME")
dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
mlflow.set_tracking_uri(tracking_uri)
model_name = "AMR-Model"
model_version = 1
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")  
# #------------------------------------------------------Making App------------------------------------------------------------------------------
from src.utils.show_recommendation import RecommendedItemsDisplay
# Selecting a user ids
st.title('Recommend Items for a User')
Select_UserID = st.selectbox("Select UserID to whom you want to recommend",user_ids)
USER_ID = Select_UserID
# Making Prediction
#rec_item = model.predict(None,(USER_ID,6))
# # Displaying
#display = RecommendedItemsDisplay(rec_item, df, USER_ID)
# show Recommendation
if st.button('Show Recommendation'):
    rec_item = model.predict((USER_ID,6))
    display = RecommendedItemsDisplay(rec_item, df, USER_ID)
    fig = display.show_recommended_images()
    st.plotly_chart(fig)
