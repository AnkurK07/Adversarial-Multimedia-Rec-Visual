'''-------------------------------------------Setting Up Streaamlit App----------------------------------------------------------------'''
import streamlit as st
import base64
# Set the background image using the external URL
page_bg_img = f"""
<style>
.stApp {{
    background-image: url("https://as2.ftcdn.net/jpg/03/66/66/93/1000_F_366669347_Po2FsQ8tV00ILhQU37sd88lQDhXHYgon.jpg");
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

# Load environment variables
load_dotenv()

DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
CONNECTION_URL = os.getenv("MONGODB_URL")

@st.cache_resource
def get_database():
    """Establish a persistent database connection."""
    client = pymongo.MongoClient(CONNECTION_URL)
    data_base = client[DB_NAME]
    return data_base

@st.cache_resource
def get_data():
    """Load and preprocess data once."""
    data_base = get_database()
    collection = data_base[COLLECTION_NAME]
    df1 = pd.DataFrame(list(collection.find()))
    df = df1.drop_duplicates(subset=["item_id", "image"])[["item_id", "image"]]
    user_ids = df1["user_id"].unique().tolist()
    return df, user_ids

# Load data once
df, user_ids = get_data()
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

st.markdown(
    "<h3 style='text-align: centre; font-size: 32px;'> Welcome to Adversial Multimedia Recommendation System For Robust Visual Recommendation ! </h3>",
    unsafe_allow_html=True
)

st.markdown(
    "<h3 style='text-align: left; font-size: 25px;'> This model recommends you visually similar items that you purchased earlier.</h3>",
    unsafe_allow_html=True
)

st.markdown(
    "<h3 style='text-align: left; font-size: 24px;'>Recommend Items for a User</h3>",
    unsafe_allow_html=True
)
Select_UserID = st.selectbox("Select UserID to whom you want to recommend",user_ids)
USER_ID = Select_UserID
if st.button('Show Recommendation'):
    rec_item = model.predict((USER_ID,6))
    display = RecommendedItemsDisplay(rec_item, df, USER_ID)
    fig = display.show_recommended_images()
    st.plotly_chart(fig)
