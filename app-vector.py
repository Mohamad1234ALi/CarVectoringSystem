import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from opensearchpy import OpenSearch
from io import StringIO
import requests
import joblib
from io import BytesIO
import boto3

# OpenSearch Configuration
OPENSEARCH_HOST = "https://search-mycarsystemdomain-3ujyrlm64nacrg4xkmoznl5oqy.us-east-1.es.amazonaws.com"
INDEX_NAME = "cars_index_new"
USERNAME = "moeuser"
PASSWORD = "Mohamad@123"


aws_access_key = st.secrets["AWS_ACCESS_KEY_ID"]
aws_secret_key = st.secrets["AWS_SECRET_ACCESS_KEY"]
aws_region = st.secrets["AWS_DEFAULT_REGION"]

boto3.setup_default_session(
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
    region_name=aws_region
)

dynamodb = boto3.resource("dynamodb", region_name="us-east-1")  # e.g. "eu-central-1"
table = dynamodb.Table("CarList")

@st.cache_resource
def load_scaler(url: str):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        scaler = joblib.load(BytesIO(response.content))
        return scaler
    except Exception as e:
        st.error(f"Failed to load scaler: {e}")
        return None

@st.cache_resource
def load_onehot_encoder(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        encoder = joblib.load(BytesIO(response.content))
        return encoder
    except Exception as e:
        st.error(f"Failed to load OneHotEncoder: {e}")
        return None
        
# Connect to OpenSearch
client = OpenSearch(
    hosts=[OPENSEARCH_HOST],
    http_auth=(USERNAME, PASSWORD),
    use_ssl=True,
    verify_certs=True,
    timeout=30
)

# Define categorical and numerical features
CATEGORICAL_FEATURES = ["BodyColor", "BodyType", "Fuel", "NumberOfDoors", "GearBox", "DriveType"]
NUMERICAL_FEATURES = ["FirstRegistration", "NumberOfSeats", "Power", "Price", "Mileage", "CubicCapacity"]


# Initialize StandardScaler
scaler_url = "https://car-recommendation-raed.s3.us-east-1.amazonaws.com/scaler/scaler.pkl"
scaler = load_scaler(scaler_url)
onehot_encoder_url = "https://car-recommendation-raed.s3.us-east-1.amazonaws.com/onehotencoder/onehot_encoder.pkl"
onehot_encoder = load_onehot_encoder(onehot_encoder_url)
# Function to convert user input into vector
def preprocess_input(category, mileage, color, doors, first_reg, gearbox, price, seats, fuel_type, performance, drivetype, cubiccapacity):
    # Prepare DataFrame for categorical input (must match training order)
    cat_input = pd.DataFrame([{
        "BodyColor": color,
        "BodyType": category,
        "Fuel": fuel_type,
        "NumberOfDoors": doors,
        "GearBox": gearbox,
        "DriveType": drivetype  # or any default, if not part of Streamlit inputs
    }])

    # OneHotEncode categorical values
    cat_encoded = onehot_encoder.transform(cat_input)

    # Scale numerical values
    numerical_scaled = scaler.transform([[first_reg, seats, performance, price, mileage, cubiccapacity]])[0]

    # Combine both parts
    return np.concatenate((cat_encoded[0], numerical_scaled))
# Function to search similar cars in OpenSearch
def search_similar_cars(query_vector):
    query = {
        "size": 10,
        "query": {
            "knn": {
                "vector": {
                    "vector": query_vector.tolist(),
                    "k": 10
                }
            }
        }
    }

    response = client.search(index=INDEX_NAME, body=query)
    return response["hits"]["hits"]


def get_car_by_id(car_id):
    response = table.get_item(Key={"CarID": car_id})
    return response.get("Item")

# Streamlit UI
st.title("Car Recommendation System 🚗")
st.write("Find similar cars 🔍")
col1, col2 = st.columns(2)

# User Inputs

with col1:
   gearbox = st.selectbox("Gearbox", onehot_encoder.categories_[CATEGORICAL_FEATURES.index("GearBox")])
    
with col2:
   gearbox_needed = st.checkbox("I need Gearbox?", value=False)

col1, col2 = st.columns(2)

with col1:
    fuel_type = st.selectbox("Fuel Type", onehot_encoder.categories_[CATEGORICAL_FEATURES.index("Fuel")])
    
with col2:
    fuel_needed = st.checkbox("I need Fuel ?",  value=False)

col1, col2 = st.columns(2)

with col1:
    category = st.selectbox("Body Type", onehot_encoder.categories_[CATEGORICAL_FEATURES.index("BodyType")])
    
with col2:
    category_needed = st.checkbox("I need Body Type ?",  value=False)

col1, col2 = st.columns(2)

with col1:
    color = st.selectbox("Body Color", onehot_encoder.categories_[CATEGORICAL_FEATURES.index("BodyColor")])
    
with col2:
    color_needed = st.checkbox("I need Body Color ?",  value=False)

col1, col2 = st.columns(2)

with col1:
    doors = st.selectbox("Number Of Doors", onehot_encoder.categories_[CATEGORICAL_FEATURES.index("NumberOfDoors")])
    
with col2:
    doors_needed = st.checkbox("I need Doors ?",  value=False)

col1, col2 = st.columns(2)

with col1:
    drivetype = st.selectbox("Drive Type", onehot_encoder.categories_[CATEGORICAL_FEATURES.index("DriveType")])
    
with col2:
    drive_needed = st.checkbox("I need Drive Type ?",  value=False)


price = st.number_input("Price ($)", min_value=1000, max_value=100000, value=5000)
mileage = st.number_input("Mileage (Km)", min_value=0, max_value=500000, value=10000)
seats = st.number_input("Number Of Seats", min_value=1, max_value=10, value=4)
performance = st.number_input("Performance", min_value=50, max_value=1000, value=100)
cubiccapacity = st.number_input("Cubic Capacity", min_value=900, max_value=4000, value=900)
first_reg = st.slider("First Registration Year", 1995, 2025, 2005)


    

if st.button("Find Similar Cars"):
    query_vector = preprocess_input(category, mileage, color, doors, first_reg, gearbox, price, seats, fuel_type, performance, drivetype, cubiccapacity)
    
    results = search_similar_cars(query_vector)
    
    if results:

        if gearbox_needed :
            results = [car for car in results if car["_source"].get("GearBox", "").lower() == gearbox.lower()]

        if fuel_needed :
            results = [car for car in results if car["_source"].get("Fuel", "").lower() == fuel_type.lower()]

        if category_needed :
            results = [car for car in results if car["_source"].get("BodyType", "").lower() == category.lower()]

        if color_needed :
            results = [car for car in results if car["_source"].get("BodyColor", "").lower() == color.lower()]

        if doors_needed :
            results = [car for car in results if car["_source"].get("NumberOfDoors", "").lower() == doors.lower()]

        if drive_needed :
            results = [car for car in results if car["_source"].get("DriveType", "").lower() == drivetype.lower()]
            
        for car in results:
            
            car_data = car["_source"]       
            real_ID = car_data["CarID"]
            full_car_info = get_car_by_id(real_ID)
        
            if full_car_info:
                st.write(f"📏 ID: {full_car_info['CarID']}  | 🔥 Body Type: {full_car_info.get('BodyType', 'N/A')} ")
                st.write(f"📏 Make: {full_car_info['Make']}  | 🔥 Model: {full_car_info.get('Model', 'N/A')} ")
                st.write(f"💡 Gearbox: {full_car_info.get('GearBox', 'N/A')} | Fuel Type : {full_car_info.get('Fuel', 'N/A')}")
                st.write(f"💡 Body Color: {full_car_info.get('BodyColor', 'N/A')} | Doors : {full_car_info.get('NumberOfDoors', 'N/A')}")
                st.write(f"💡 Drive Type: {full_car_info.get('DriveType', 'N/A')} | Mileage : {full_car_info.get('Mileage', 'N/A')}")
                st.write(f"📅 First Registration: {full_car_info.get('FirstRegistration', 'N/A')} | 💰 Price: {full_car_info.get('Price', 'N/A')}")
                st.write("---")
            else:
                 st.write(f"❌ Car with ID {real_ID} not found in DynamoDB.")
    else:
        st.write("❌ No similar cars found.")
