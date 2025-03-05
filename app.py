'''-------------------------------------------Setting Up Streaamlit App----------------------------------------------------------------'''
import streamlit as st
import base64
# Set the background image using the external URL
page_bg_img = f"""
<style>
.stApp {{
    background-image: url("https://static.vecteezy.com/system/resources/previews/035/701/136/non_2x/set-of-fashion-clothes-for-women-casual-garments-and-accessories-for-spring-and-summer-jacket-bags-shoes-trousers-dress-hats-flying-flat-illustrations-isolated-on-white-background-vector.jpg");
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
model_name = "AMR-Model"
model_version = 1

@st.cache_resource
def load_model():
    """Load the model only once and cache it."""
    dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
    mlflow.set_tracking_uri(tracking_uri)
    model_name = "AMR-Model"
    model_version = 1
    print("Loading model for the first time...")
    return mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")

# Load model once and reuse
model = load_model()
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
