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
OPENSEARCH_HOST = "https://search-mydomainsystemnew-dbwxhhxv6gagjbddxbu5wxejou.us-east-1.es.amazonaws.com"
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
def load_label_encoders(urls: dict):
    """Loads LabelEncoders from S3 URLs and returns a dictionary of encoders."""
    encoders = {}
    for feature, url in urls.items():
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for HTTP errors
            encoders[feature] = joblib.load(BytesIO(response.content))
        except Exception as e:
            st.error(f"Failed to load {feature} encoder: {e}")
            encoders[feature] = None
    return encoders


# Connect to OpenSearch
client = OpenSearch(
    hosts=[OPENSEARCH_HOST],
    http_auth=(USERNAME, PASSWORD),
    use_ssl=True,
    verify_certs=True,
    timeout=30
)

# Define categorical and numerical features
CATEGORICAL_FEATURES = ["BodyColor", "BodyType", "Fuel", "NumberOfDoors", "GearBox"]
NUMERICAL_FEATURES = ["FirstRegistration", "NumberOfSeats", "Power", "Price", "Mileage"]

# URLs for LabelEncoders in S3
label_encoder_urls = {
    "BodyColor": "https://car-recommendation-raed.s3.us-east-1.amazonaws.com/label_encoders/BodyColor_encoder.pkl",
    "BodyType": "https://car-recommendation-raed.s3.us-east-1.amazonaws.com/label_encoders/BodyType_encoder.pkl",
    "Fuel": "https://car-recommendation-raed.s3.us-east-1.amazonaws.com/label_encoders/Fuel_encoder.pkl",
    "NumberOfDoors": "https://car-recommendation-raed.s3.us-east-1.amazonaws.com/label_encoders/NumberOfDoors_encoder.pkl",
    "GearBox": "https://car-recommendation-raed.s3.us-east-1.amazonaws.com/label_encoders/GearBox_encoder.pkl",
}

label_encoders = load_label_encoders(label_encoder_urls)

# Initialize StandardScaler
scaler_url = "https://car-recommendation-raed.s3.us-east-1.amazonaws.com/scaler/scaler.pkl"
scaler = load_scaler(scaler_url)
    
# Function to convert user input into vector
def preprocess_input(category, mileage, color, doors, first_reg, gearbox, price, seats, fuel_type, performance):

    
    BodyColor_encoded = label_encoders["BodyColor"].transform([color])[0]
    BodyType_encoded = label_encoders["BodyType"].transform([category])[0]
    Fuel_encoded = label_encoders["Fuel"].transform([fuel_type])[0]
    NumberOfDoors_encoded = label_encoders["NumberOfDoors"].transform([doors])[0]
    GearBox_encoded = label_encoders["GearBox"].transform([gearbox])[0]

    # numerical_values = np.array([[first_reg, price, mileage, performance]])
    # numerical_scaled = scaler.transform(numerical_values)[0] 
    numerical_scaled = scaler.transform([[first_reg, seats, performance, price, mileage]])[0]  # Flatten the result

    return np.concatenate(([BodyColor_encoded, BodyType_encoded, Fuel_encoded, NumberOfDoors_encoded, GearBox_encoded], numerical_scaled))

# Function to search similar cars in OpenSearch
def search_similar_cars(query_vector):
    query = {
        "size": 50,
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
colf1, colf2 = st.columns(2)

# User Inputs
category = st.selectbox("Body Type", label_encoders["BodyType"].classes_)
#accident = st.selectbox("Accident Free:", label_encoders["AccidentFree"].classes_)
color = st.selectbox("Body Color", label_encoders["BodyColor"].classes_)
doors = st.selectbox("Number Of Doors", label_encoders["NumberOfDoors"].classes_)
first_reg = st.slider("First Registration Year", 1980, 2025, 2005)

with col1:
   gearbox = st.selectbox("Gearbox", label_encoders["GearBox"].classes_)

with col2:
   gearbox_needed = st.checkbox("I need Gearbox?", value=False)
  
    
    
price = st.number_input("Price ($)", min_value=1000, max_value=100000, value=5000)
mileage = st.number_input("Mileage (Km)", min_value=0, max_value=500000, value=10000)
seats = st.number_input("Number Of Seats", min_value=1, max_value=10, value=4)

with colf1:
    fuel_type = st.selectbox("Fuel Type", label_encoders["Fuel"].classes_)

with colf2:
    fuel_needed = st.checkbox("I need Fuel ?",  value=False)
    
performance = st.number_input("Performance", min_value=50, max_value=1000, value=100)

if st.button("Find Similar Cars"):
    query_vector = preprocess_input(category, mileage, color, doors, first_reg, gearbox, price, seats, fuel_type, performance)
    
    results = search_similar_cars(query_vector)
    
    if results:

        if gearbox_needed :
            results = [car for car in results if car["_source"].get("GearBox", "").lower() == gearbox.lower()]

        if fuel_needed :
            results = [car for car in results if car["_source"].get("Fuel", "").lower() == fuel_type.lower()]
            
        for car in results:
            
            car_data = car["_source"]       
            real_ID = car_data["CarID"]
            full_car_info = get_car_by_id(real_ID)
        
            if full_car_info:
                st.write(f"📏 ID: {full_car_info['CarID']}  | 🔥 Type: {full_car_info.get('BodyType', 'N/A')} ")
                st.write(f"💡 Gearbox: {full_car_info.get('GearBox', 'N/A')} | Fuel Type: {full_car_info.get('Fuel', 'N/A')}")
                st.write(f"📅 First Registration: {full_car_info.get('FirstRegistration', 'N/A')} | 💰 Price: {full_car_info.get('Price', 'N/A')}")
                st.write("---")
            else:
                 st.write(f"❌ Car with ID {real_ID} not found in DynamoDB.")
    else:
        st.write("❌ No similar cars found.")
