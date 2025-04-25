import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from opensearchpy import OpenSearch
from io import StringIO
import requests
import joblib
from io import BytesIO
# OpenSearch Configuration
OPENSEARCH_HOST = "https://search-carsystemvector-vsbohud2ogupvyw6ssdoddybgm.us-east-1.es.amazonaws.com"
INDEX_NAME = "cars_index_new"
USERNAME = "moeuser"
PASSWORD = "Mohamad@123"

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
CATEGORICAL_FEATURES = ["AccidentFree", "BodyColor", "BodyType", "Fuel", "NumberOfDoors"]
NUMERICAL_FEATURES = ["FirstRegistration", "NumberOfSeats", "Power", "Price"]

# URLs for LabelEncoders in S3
label_encoder_urls = {
    "AccidentFree": "https://car-recommendation-raed.s3.us-east-1.amazonaws.com/label_encoders/AccidentFree_encoder.pkl",
    "BodyColor": "https://car-recommendation-raed.s3.us-east-1.amazonaws.com/label_encoders/BodyColor_encoder.pkl",
    "BodyType": "https://car-recommendation-raed.s3.us-east-1.amazonaws.com/label_encoders/BodyType_encoder.pkl",
    "Fuel": "https://car-recommendation-raed.s3.us-east-1.amazonaws.com/label_encoders/Fuel_encoder.pkl",
    "NumberOfDoors": "https://car-recommendation-raed.s3.us-east-1.amazonaws.com/label_encoders/NumberOfDoors_encoder.pkl",
}

label_encoders = load_label_encoders(label_encoder_urls)

# Initialize StandardScaler
scaler_url = "https://car-recommendation-raed.s3.us-east-1.amazonaws.com/scaler/scaler.pkl"
scaler = load_scaler(scaler_url)
    
# Function to convert user input into vector
def preprocess_input(category, accident, color, doors, first_reg, gearbox, price, seats, fuel_type, performance):

    AccidentFree_encoded = label_encoders["AccidentFree"].transform([accident])[0]
    BodyColor_encoded = label_encoders["BodyColor"].transform([color])[0]
    BodyType_encoded = label_encoders["BodyType"].transform([category])[0]
    Fuel_encoded = label_encoders["Fuel"].transform([fuel_type])[0]
    NumberOfDoors_encoded = label_encoders["NumberOfDoors"].transform([doors])[0]

    # numerical_values = np.array([[first_reg, price, mileage, performance]])
    # numerical_scaled = scaler.transform(numerical_values)[0] 
    numerical_scaled = scaler.transform([[first_reg, seats, performance, price]])[0]  # Flatten the result

    return np.concatenate(([AccidentFree_encoded, BodyColor_encoded, BodyType_encoded, Fuel_encoded, NumberOfDoors_encoded], numerical_scaled))

# Function to search similar cars in OpenSearch
def search_similar_cars(query_vector):
    query = {
        "size": 9,
        "query": {
            "knn": {
                "vector": {
                    "vector": query_vector.tolist(),
                    "k": 9
                }
            }
        }
    }

    response = client.search(index=INDEX_NAME, body=query)
    return response["hits"]["hits"]


# Streamlit UI
st.title("Car Recommendation System 🚗")
st.write("Find similar cars 🔍")

# User Inputs
category = st.selectbox("Body Type", label_encoders["BodyType"].classes_)
accident = st.selectbox("Accident Free:", label_encoders["AccidentFree"].classes_)
color = st.selectbox("Body Color", label_encoders["BodyColor"].classes_)
doors = st.selectbox("Number Of Doors", label_encoders["NumberOfDoors"].classes_)
first_reg = st.slider("First Registration Year", 2000, 2025, 2015)
gearbox = st.selectbox("Gearbox", ["Any", "Manual", "Semiautomatic", "Automatic"])
price = st.number_input("Price ($)", min_value=1000, max_value=100000, value=20000)
seats = st.number_input("Number Of Seats", min_value=1, max_value=10, value=1)
fuel_type = st.selectbox("Fuel Type", label_encoders["Fuel"].classes_)
performance = st.number_input("Performance", min_value=50, max_value=1000, value=150)

if st.button("Find Similar Cars"):
    query_vector = preprocess_input(category, accident, color, doors, first_reg, gearbox, price, seats, fuel_type, performance)
    
    results = search_similar_cars(query_vector)
    
    if results:
        for car in results:
            car_data = car["_source"]
            
            real_ID = car_data["CarID"]
            real_Type = car_data["BodyType"]
            real_gearbox = car_data["GearBox"]
            real_Fuel = car_data["Fuel"]
  


            
          
            st.write(f"📏 ID: {real_ID}  | 🔥 Type: {real_Type} ")
            st.write(f"💡  Gearbox: {real_gearbox} | Fuel Type: {real_Fuel}")
            st.write("---")
    else:
        st.write("❌ No similar cars found.")
