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
OPENSEARCH_HOST = "https://search-mycarsystemdomain-3ujyrlm64nacrg4xkmoznl5oqy.us-east-1.es.amazonaws.com" # the opensearch endpoint
INDEX_NAME = "cars_index_new"
USERNAME = "moeuser"
PASSWORD = "Mohamad@123"

# Get the parameters from the settings in streamlit app
aws_access_key = st.secrets["AWS_ACCESS_KEY_ID"]
aws_secret_key = st.secrets["AWS_SECRET_ACCESS_KEY"]
aws_region = st.secrets["AWS_DEFAULT_REGION"]

# Initialize the boto3
boto3.setup_default_session(
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
    region_name=aws_region
)

# Connect and access to the dynamodb table 
dynamodb = boto3.resource("dynamodb", region_name="us-east-1") 
table = dynamodb.Table("CarList")

# Load scaler from s3 (we need the same scaler while scale the numerical data (systemvectorizaion file))
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
        
# Load onehotencoder from s3 (we need the same encoder while encode the categorical data (systemvectorizaion file))
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
CATEGORICAL_FEATURES = ["BodyType", "Fuel", "NumberOfDoors", "GearBox", "DriveType", "NumberOfSeats"]
NUMERICAL_FEATURES = ["FirstRegistration", "Power", "CubicCapacity"]


# Initialize StandardScaler and OneHotEncoder
scaler_url = "https://car-recommendation-raed.s3.us-east-1.amazonaws.com/scaler/scaler.pkl"
scaler = load_scaler(scaler_url)

onehot_encoder_url = "https://car-recommendation-raed.s3.us-east-1.amazonaws.com/onehotencoder/onehot_encoder.pkl"
onehot_encoder = load_onehot_encoder(onehot_encoder_url)

# Function to convert user input into vector
def preprocess_input(category, doors, first_reg, gearbox, seats, fuel_type, performance, drivetype, cubiccapacity):
    # Prepare DataFrame for categorical input (must match training order)
    cat_input = pd.DataFrame([{
        "BodyType": category,
        "Fuel": fuel_type,
        "NumberOfDoors": doors,
        "GearBox": gearbox,
        "DriveType": drivetype,  # or any default, if not part of Streamlit inputs
        "NumberOfSeats": seats
    }])

    # OneHotEncode categorical values
    cat_encoded = onehot_encoder.transform(cat_input)

   # Apply log1p to numerical features
    numerical_input = [first_reg, performance, cubiccapacity]
    log_transformed = np.log1p(numerical_input).reshape(1, -1)

    # Apply scaler
    numerical_scaled = scaler.transform(log_transformed)[0]

    # Combine categorical + numerical
    return np.concatenate((cat_encoded[0], numerical_scaled))
    
# Function to search similar cars in OpenSearch
def search_similar_cars(query_vector,similarity_threshold=0.8):
    query = {
        "size": 50, # how many result showing to user (size <= k)
        "query": {
            "knn": {
                "vector": {
                    "vector": query_vector.tolist(),
                    "k": 50  #  how many k search in the index (table)
                }
            }
        }
    }

    response = client.search(index=INDEX_NAME, body=query)
    results = response["hits"]["hits"]

    # Filter by similarity threshold
    filtered = [r for r in results if 0.6 <= r["_score"] <= similarity_threshold]
    return filtered

# Function for get the ad from the dynamodb by ID
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
    doors = st.selectbox("Number Of Doors", onehot_encoder.categories_[CATEGORICAL_FEATURES.index("NumberOfDoors")])
    
with col2:
    doors_needed = st.checkbox("I need Doors ?",  value=False)

col1, col2 = st.columns(2)

with col1:
    drivetype = st.selectbox("Drive Type", onehot_encoder.categories_[CATEGORICAL_FEATURES.index("DriveType")])
    
with col2:
    drive_needed = st.checkbox("I need Drive Type ?",  value=False)

col1, col2 = st.columns(2)

with col1:
    seats = st.selectbox("Number Of Seats", onehot_encoder.categories_[CATEGORICAL_FEATURES.index("NumberOfSeats")])
    
with col2:
    seats_needed = st.checkbox("I need Number Of Seats ?",  value=False)


price_range = st.slider(
    "Price Range ($)", 
    min_value=500, 
    max_value=150000, 
    value=(5000, 30000),  # default range
    step=1000
)
mileage_range = st.slider(
    "Mileage Range (Km)", 
    min_value=0, 
    max_value=500000, 
    value=(0, 100000),  # default range
    step=1000
)
performance = st.number_input("Performance", min_value=50, max_value=1000, value=100)
cubiccapacity = st.number_input("Cubic Capacity", min_value=900, max_value=4000, value=900)
first_reg = st.slider("First Registration Year", 1995, 2025, 2005)

price_min, price_max = price_range
mileage_min, mileage_max = mileage_range
    

if st.button("Find Similar Cars"):
    query_vector = preprocess_input(category, doors, first_reg, gearbox, seats, fuel_type, performance, drivetype, cubiccapacity)
    
    results = search_similar_cars(query_vector, similarity_threshold=0.8)
    count = len(results)
    st.write(f"🔍 Found {count} similar cars")
    
    if results:
        # Filtering the data depends on the choice of the user
        if gearbox_needed :
            results = [car for car in results if car["_source"].get("GearBox", "").lower() == gearbox.lower()]

        if fuel_needed :
            results = [car for car in results if car["_source"].get("Fuel", "").lower() == fuel_type.lower()]

        if category_needed :
            results = [car for car in results if car["_source"].get("BodyType", "").lower() == category.lower()]

        if doors_needed :
            results = [car for car in results if car["_source"].get("NumberOfDoors", "").lower() == doors.lower()]

        if drive_needed :
            results = [car for car in results if car["_source"].get("DriveType", "").lower() == drivetype.lower()]

        if seats_needed :
            results = [car for car in results if car["_source"].get("NumberOfSeats", "") == seats]

        # Filter by Price range
       # results = [
            #car for car in results
            #if price_min <= int(car["_source"].get("Price", 0)) <= price_max
        #]

        # Filter by Mileage range
       # results = [
            #car for car in results
           # if mileage_min <= int(car["_source"].get("Mileage", 0)) <= mileage_max
       # ]
       
        
        for car in results:
            
            car_data = car["_source"]       
            real_ID = car_data["CarID"]
            full_car_info = get_car_by_id(real_ID)
        
            if full_car_info:
                st.write(f"🆔 ID: {full_car_info['CarID']}  | 🔥 Body Type: {full_car_info.get('BodyType', 'N/A')} ")
                st.write(f"📏 Make: {full_car_info['Make']}  | 📏 Model: {full_car_info.get('Model', 'N/A')} ")
                st.write(f"⚙️ Gearbox: {full_car_info.get('GearBox', 'N/A')} | ⛽ Fuel Type : {full_car_info.get('Fuel', 'N/A')}")
                st.write(f"💡 Body Color: {full_car_info.get('BodyColor', 'N/A')} | 🚪 Doors : {full_car_info.get('NumberOfDoors', 'N/A')}")
                st.write(f"🚙 Drive Type: {full_car_info.get('DriveType', 'N/A')} | 🚗📏 Mileage : {full_car_info.get('Mileage', 'N/A')}")
                st.write(f"🏁 Cubic Capacity: {full_car_info.get('CubicCapacity', 'N/A')} | ⚡ Performance : {full_car_info.get('Power', 'N/A')}")
                st.write(f"👥 Number Of Seats: {full_car_info.get('NumberOfSeats', 'N/A')} | 🛠️ Usage State : {full_car_info.get('UsageState', 'N/A')}")
                st.write(f"📅 First Registration: {full_car_info.get('FirstRegistration', 'N/A')} | 💰 Price: {full_car_info.get('Price', 'N/A')}")
                st.write(f"📅 Score : {car['_score']}")
                st.write("---")
            else:
                 st.write(f"❌ Car with ID {real_ID} not found in DynamoDB.")
    else:
        st.write("❌ No similar cars found.")
