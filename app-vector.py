import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from opensearchpy import OpenSearch
from io import StringIO
import requests
import joblib
from io import BytesIO
import boto3
import random
import re
import json

# OpenSearch Configuration
OPENSEARCH_HOST = st.secrets["OPENSEARCH_HOST"] # the opensearch endpoint
INDEX_NAME = "cars_index_new"
USERNAME = st.secrets["USERNAME"]
PASSWORD = st.secrets["PASSWORD"]

# Get the parameters from the settings in streamlit app
aws_access_key = st.secrets["AWS_ACCESS_KEY_ID"]
aws_secret_key = st.secrets["AWS_SECRET_ACCESS_KEY"]
aws_region = st.secrets["AWS_DEFAULT_REGION"]

api_key = st.secrets["api_key"]
endpoint = st.secrets["endpoint"]
deployment_name = st.secrets["deployment_name"]
api_version = st.secrets["api_version"]


# Initialize the boto3
boto3.setup_default_session(
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
    region_name=aws_region
)

# Connect and access to the dynamodb table 
dynamodb = boto3.resource(st.secrets["DYNAMODB"], region_name="us-east-1") 
table = dynamodb.Table(st.secrets["DYNAMO_TABLE"])

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
scaler_url = st.secrets["SCALER_URL"]
scaler = load_scaler(scaler_url)


url = st.secrets["CATEGORIES_URL"]

response = requests.get(url)
response.raise_for_status()  # to catch HTTP errors

categories_list = response.json()  # load JSON content directly


dummy_input = pd.DataFrame(
    list(zip(*categories_list)),  # transpose so each row has one value from each feature
    columns=CATEGORICAL_FEATURES
)

# Now fit the encoder
onehot_encoder = OneHotEncoder(categories=categories_list, handle_unknown='ignore', sparse_output=False)
onehot_encoder.fit(dummy_input)





# Function to convert user input into vector
def preprocess_input(category, doors, first_reg, gearbox, seats, fuel_type, performance, drivetype, cubiccapacity):
  
    cat_data = {
    "BodyType": None if category == "Any" else category,
    "Fuel": None if fuel_type == "Any" else fuel_type,
    "NumberOfDoors": None if doors == "Any" else doors,
    "GearBox": None if gearbox == "Any" else gearbox,
    "DriveType": None if drivetype == "Any" else drivetype,
    "NumberOfSeats": None if seats == "Any" else seats
    }

    # OneHotEncoder should be fitted with handle_unknown='ignore'
    cat_df = pd.DataFrame([cat_data])
    cat_encoded = onehot_encoder.transform(cat_df)

   # Apply log1p to numerical features
    
    numerical_input = [
        0 if first_reg in ["Any", None] else first_reg,
        0 if performance in ["Any", None] else performance,
        0 if cubiccapacity in ["Any", None] else cubiccapacity
    ]
    log_transformed = np.log1p(numerical_input).reshape(1, -1)

    # Apply scaler
    numerical_scaled = scaler.transform(log_transformed)[0]
    
    combined = np.concatenate((cat_encoded[0], numerical_scaled))
    normalized = combined / np.linalg.norm(combined)
    return normalized
    
    
    
# Function to search similar cars in OpenSearch

def search_similar_cars_with_filters(
    query_vector, 
    numberofcars, 
    price_min, price_max, 
    mileage_min,mileage_max, 
    similarity_threshold,
    gearbox_needed,fuel_needed,
    category_needed,doors_needed,
    drive_needed,seats_needed,
    gearbox_value,fuel_value,
    category_value,doors_value,
    drive_value,seats_value
):
  
    # Build the range filters list
    filters = []
    if price_min is not None or price_max is not None:
        price_range = {}
        if price_min is not None:
            price_range["gte"] = price_min
        if price_max is not None:
            price_range["lte"] = price_max
        filters.append({"range": {"Price": price_range}})

    if mileage_min is not None or mileage_max is not None:
        mileage_range = {}
        if mileage_min is not None:
            mileage_range["gte"] = mileage_min
        if mileage_max is not None:
            mileage_range["lte"] = mileage_max
        filters.append({"range": {"Mileage": mileage_range}})

    if gearbox_needed and gearbox_value != "Any":
        filters.append({"term": {"GearBox": gearbox_value}})

    if fuel_needed and fuel_value != "Any":
        filters.append({"term": {"Fuel": fuel_value}})

    if category_needed and category_value != "Any":
        filters.append({"term": {"BodyType": category_value}})

    if doors_needed and doors_value != "Any":
        filters.append({"term": {"NumberOfDoors": doors_value}})

    if drive_needed and drive_value != "Any":
        filters.append({"term": {"DriveType": drive_value}})

    if seats_needed and seats_value != "Any":
        filters.append({"term": {"NumberOfSeats": seats_value}})

    # Construct the query with bool filter and knn must
    query = {
     "size": numberofcars * 10,
     "query": {
        "knn": {
            "vector": {
                "vector": query_vector.tolist(),
                "k": numberofcars * 10,
                "filter": {
                    "bool": {
                        "must": filters
                    }
                }
            }
        }
      }
    }


    
    # Execute the search
    response = client.search(index=INDEX_NAME, body=query)
    results = response["hits"]["hits"]
    count_result = len(results)

    # Optional: filter results by similarity threshold on _score
    filtered = [r for r in results if r["_score"] >= similarity_threshold]
    random.shuffle(filtered)
    return filtered[:numberofcars], count_result
  
def search_similar_cars_without_filters(
    query_vector, 
    numberofcars, 
    similarity_threshold,
    
):
  
    # Construct the query with bool filter and knn must
    query = {
     "size": numberofcars * 10,
     "query": {
        "knn": {
            "vector": {
                "vector": query_vector.tolist(),
                "k": numberofcars * 10
            }
        }
      }
    }


    
    # Execute the search
    response = client.search(index=INDEX_NAME, body=query)
    results = response["hits"]["hits"]
    count_result = len(results)

    # Optional: filter results by similarity threshold on _score
    filtered = [r for r in results if r["_score"] >= similarity_threshold]
    random.shuffle(filtered)
    return filtered[:numberofcars], count_result


def search_count_Filter(
    client,
    index_name,
    price_min, price_max, 
    mileage_min,mileage_max,
    gearbox_needed,fuel_needed,
    category_needed,doors_needed,
    drive_needed,seats_needed,
    gearbox_value,fuel_value,
    category_value,doors_value,
    drive_value,seats_value
):
    # Build filter conditions
    filters = []

    if price_min is not None or price_max is not None:
        price_range = {}
        if price_min is not None:
            price_range["gte"] = price_min
        if price_max is not None:
            price_range["lte"] = price_max
        filters.append({"range": {"Price": price_range}})

    if mileage_min is not None or mileage_max is not None:
        mileage_range = {}
        if mileage_min is not None:
            mileage_range["gte"] = mileage_min
        if mileage_max is not None:
            mileage_range["lte"] = mileage_max
        filters.append({"range": {"Mileage": mileage_range}})

    if gearbox_needed and gearbox_value != "Any":
        filters.append({"term": {"GearBox": gearbox_value}})

    if fuel_needed and fuel_value != "Any":
        filters.append({"term": {"Fuel": fuel_value}})

    if category_needed and category_value != "Any":
        filters.append({"term": {"BodyType": category_value}})

    if doors_needed and doors_value != "Any":
        filters.append({"term": {"NumberOfDoors": doors_value}})

    if drive_needed and drive_value != "Any":
        filters.append({"term": {"DriveType": drive_value}})

    if seats_needed and seats_value != "Any":
        filters.append({"term": {"NumberOfSeats": seats_value}})

    # 🔢 Count how many cars match the filters only (before KNN)
    count_query = {
        "query": {
            "bool": {
                "filter": filters
            }
        }
    }
    filter_count = client.count(index=INDEX_NAME, body=count_query)["count"]

    
    return filter_count


# Function for get the ad from the dynamodb by ID
def get_car_by_id(car_id):
    response = table.get_item(Key={"CarID": car_id})
    return response.get("Item")

# Streamlit UI
st.title("Car Recommendation System 🚗")
st.write("Find similar cars 🔍")


numberofcars = st.number_input("Number of cars to be searched", min_value=5, max_value=5000, value=10)

percentage = st.number_input("Similarity Percentage", min_value=10, max_value=100, value=60)
percentagefinal = percentage / 100

# User Inputs


col1, col2 = st.columns(2)

with col1:
   gearbox = st.selectbox("Gearbox", ["Any"] + categories_list[CATEGORICAL_FEATURES.index("GearBox")])
    
with col2:
   st.markdown("######")
   gearbox_needed = st.checkbox("Filter by Gearbox", value=False)

col1, col2 = st.columns(2)

with col1:
    fuel_type = st.selectbox("Fuel Type", ["Any"] + categories_list[CATEGORICAL_FEATURES.index("Fuel")])
    
with col2:
    st.markdown("######")
    fuel_needed = st.checkbox("Filter by Fuel",  value=False)

col1, col2 = st.columns(2)

with col1:
    category = st.selectbox("Body Type", ["Any"] + categories_list[CATEGORICAL_FEATURES.index("BodyType")])
    
with col2:
    st.markdown("######")
    category_needed = st.checkbox("Filter by Body Type",  value=False)

col1, col2 = st.columns(2)

with col1:
    doors = st.selectbox("Number Of Doors", ["Any"] + categories_list[CATEGORICAL_FEATURES.index("NumberOfDoors")])
    
with col2:
    st.markdown("######")
    doors_needed = st.checkbox("Filter by Doors",  value=False)

col1, col2 = st.columns(2)

with col1:
    drivetype = st.selectbox("Drive Type", ["Any"] + categories_list[CATEGORICAL_FEATURES.index("DriveType")])
    
with col2:
    st.markdown("######")
    drive_needed = st.checkbox("Filter by Drive Type",  value=False)

col1, col2 = st.columns(2)

with col1:
    seats = st.selectbox("Number Of Seats", ["Any"] + categories_list[CATEGORICAL_FEATURES.index("NumberOfSeats")])
    
with col2:
    st.markdown("######")
    seats_needed = st.checkbox("Filter by Seats",  value=False)

    
performance = st.number_input("Performance", min_value=50, max_value=1000, value=100)
cubiccapacity = st.number_input("Cubic Capacity", min_value=900, max_value=4000, value=900)


price_range = st.slider(
    "Price Range ($)", 
    min_value=500, 
    max_value=150000, 
    value=(2000, 30000),  # default range
    step=1000
)
mileage_range = st.slider(
    "Mileage Range (Km)", 
    min_value=0, 
    max_value=500000, 
    value=(0, 150000),  # default range
    step=1000
)
first_reg = st.slider("First Registration Year", 1995, 2025, 2005)
price_min, price_max = price_range
mileage_min, mileage_max = mileage_range
    
if st.button("Find Similar Cars") :

    count = search_count_Filter(
      client=client,
      index_name=INDEX_NAME,
      price_min=price_min,
      price_max=price_max,
      mileage_min=mileage_min,
      mileage_max=mileage_max,
      gearbox_needed=gearbox_needed , 
      fuel_needed=fuel_needed,category_needed=category_needed,doors_needed=doors_needed,drive_needed=drive_needed,
      seats_needed=seats_needed , gearbox_value=gearbox ,fuel_value=fuel_type, category_value=category,doors_value=doors,
      drive_value=drivetype,seats_value=seats
    )
    st.write(f"🧮 {price_min} and {price_max} the price range.")
    st.write(f"🧮 {mileage_min} and {mileage_max} the mileage range.")
    
    st.write(f"🧮 {count} cars match your filter criteria.")
   
    query_vector = preprocess_input(category, doors, first_reg, gearbox, seats, fuel_type, performance, drivetype, cubiccapacity)
    
    results, count_results = search_similar_cars_with_filters(query_vector,numberofcars,price_min,price_max,mileage_min,mileage_max, similarity_threshold=percentagefinal,gearbox_needed=gearbox_needed , 
                                                             fuel_needed=fuel_needed,category_needed=category_needed,doors_needed=doors_needed,drive_needed=drive_needed,
                                                             seats_needed=seats_needed , gearbox_value=gearbox ,fuel_value=fuel_type, category_value=category,doors_value=doors
                                                              , drive_value=drivetype,seats_value=seats)
    count = len(results)
    st.markdown("<br>", unsafe_allow_html=True)
    st.write(f"🔍 Found {count_results} similar cars using cosine")
    st.write(f"🔍 Found {count} similar cars after filtering with percentage {percentagefinal}")
    
    if results:
        # Filtering the data depends on the choice of the user
        st.markdown("<br>", unsafe_allow_html=True)
        for car in results:
            
            car_data = car["_source"]       
            real_ID = car_data["CarID"]
            full_car_info = get_car_by_id(real_ID)
        
            if full_car_info:
                st.write(f"🆔 ID: {full_car_info['CarID']}")
                st.write(f"🔥 Body Type: {full_car_info.get('BodyType', 'N/A')}")
                st.write(f"📏 Make: {full_car_info['Make']}  | 📏 Model: {full_car_info.get('Model', 'N/A')} ")
                st.write(f"⚙️ Gearbox: {full_car_info.get('GearBox', 'N/A')} | ⛽ Fuel Type : {full_car_info.get('Fuel', 'N/A')}")
                st.write(f"💡 Body Color: {full_car_info.get('BodyColor', 'N/A')} | 🚪 Doors : {full_car_info.get('NumberOfDoors', 'N/A')}")
                st.write(f"🚙 Drive Type: {full_car_info.get('DriveType', 'N/A')} | 🚗📏 Mileage : {full_car_info.get('Mileage', 'N/A')}")
                st.write(f"🏁 Cubic Capacity: {full_car_info.get('CubicCapacity', 'N/A')} | ⚡ Performance : {full_car_info.get('Power', 'N/A')}")
                st.write(f"👥 Number Of Seats: {full_car_info.get('NumberOfSeats', 'N/A')} | 🛠️ Usage State : {full_car_info.get('UsageState', 'N/A')}")
                st.write(f"📅 First Registration: {full_car_info.get('FirstRegistration', 'N/A')} | 💰 Price: {full_car_info.get('Price', 'N/A')}")
                #st.write(f"📅 Score : {car['_score']}")
                st.write("---")
            else:
                 st.write(f"❌ Car with ID {real_ID} not found in DynamoDB.")
    else:
        st.write("❌ No similar cars found.")



def user_says_rest_doesnt_matter(user_msg):
    user_msg = user_msg.lower()
    phrases = [
        "i don't care about the rest",
        "i don't care about the others",
        "alles andere ist mir egal",
        "mir ist der rest egal",
        "rest doesn't matter",
        "the rest doesn't matter",
        "everything else is fine",
        "whatever else",
        "egal was sonst"
    ]
    return any(p in user_msg for p in phrases)

def handle_any_statements(json_text, user_message, current_prefs=None):
    """
    Updates parsed JSON response to handle 'any' logic for categorical fields.

    - If the user says something like "I don't care about the others" → sets all missing categorical fields to "any".
    - If the user says "any", "egal", or similar → sets missing categorical field(s) to "any".
    - Applies fallback when only one categorical field is missing and user says "any".
    """

    CATEGORICAL_FIELDS = {
        "gearbox",
        "fueltype",
        "bodytype",
        "numberOfDoors",
        "driveType"
    }

    try:
        parsed = json.loads(json_text)
    except Exception:
        return None  # If it's not valid JSON, exit

    user_msg_lower = user_message.strip().lower()
    any_phrases = ["any", "egal", "whatever", "doesn't matter", "don't care", "no preference", "kein unterschied", "macht nichts", "egal ist mir"]

    # Global 'any': apply to all missing categorical fields
    if any(phrase in user_msg_lower for phrase in ["i don't care about the others", "i don't care about the rest", "mir egal was der rest ist", "alles andere ist mir egal"]):
        for field in CATEGORICAL_FIELDS:
            if (current_prefs is None or current_prefs.get(field) is None) and parsed.get(field) is None:
                parsed[field] = "any"

    # Field-specific 'any' detection (if user refers to a field directly)
    for field in CATEGORICAL_FIELDS:
        if field in user_msg_lower or field.lower().replace("_", " ") in user_msg_lower:
            if any(phrase in user_msg_lower for phrase in any_phrases):
                parsed[field] = "any"

    # Fallback: if user says only "any" and exactly one categorical field is missing
    if any(user_msg_lower.strip() == phrase for phrase in any_phrases):
        if current_prefs:
            missing = [
                field for field in CATEGORICAL_FIELDS
                if current_prefs.get(field) is None and parsed.get(field) is None
            ]
            if len(missing) == 1:
                parsed[missing[0]] = "any"

    return parsed


def render_chat_history():
    for msg in reversed(st.session_state.chat_history):
        if msg["role"] == "user":
            st.markdown(f"**🧑 You:** {msg['content']}")
        elif msg["role"] == "assistant":
            st.markdown(f"**🤖 Assistant:** {msg['content']}")


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.awaiting_followup = False
    st.session_state.current_preferences = {}
    

# Build follow-up prompt dynamically from preferences and missing fields
def build_follow_up_prompt(prefs, missing_fields, last_user_message=""):
    json_formatted = json.dumps(prefs, indent=2)
   

    return f"""
You are helping a user find a suitable used car.

Based on the previous message, we extracted the following preferences:

{json_formatted}

Some values are still missing: {", ".join(missing_fields)}.

The user just wrote: \"{last_user_message}\".

🎯 Your task:
→ If the user clearly gives one of the missing values, return a new valid JSON object with only that update.
→ If the user sounds unsure or confused, respond in natural language. Do NOT return JSON in that case.

If they seem unsure or ask for help (e.g. “Ich weiß nicht”, “Hilf mir”, “Hilfe”, “Help me”), do NOT repeat the same question.

Instead:
- Briefly explain what the missing value means in simple, friendly language
- Then ask the question again in an easier and more helpful way

🛑 Avoid:
- technical field names like 'gearbox', 'performance_kw'
- car terms like 'cubic capacity', 'numberOfDoors'

✅ Instead:
- Use everyday language like:
  - “Do you usually drive alone or with others?”
  - “How old can the car be at most?”

Respond in the same language the user used.
""".strip()

# System prompts

def get_system_prompt(phase, last_user_message=""):
    if phase == "initial":
        return r'''You are a smart and helpful car assistant. Your task is to understand what kind of car the user is looking for, based on natural language.

You will extract their wishes and return them as a valid JSON object using the format and allowed values below.


🎯 Your goal is to fill all fields that can be reasonably inferred. Use the exact terms shown. If something is unclear or missing, set it to `null`.

❗ If the user explicitly says they do NOT want something (e.g. "no limousine", "not electric", "not diesel"),  
✅ then set the corresponding value to `null` unless another valid alternative is clearly preferred.

Allowed values and structure:

{{
  "gearbox": "AUTOMATIC" | "MANUAL" | "SEMI_AUTOMATIC",
  "fueltype": "CNG" | "DIESEL" | "ELECTRICITY" | "ETHANOL" | "HYBRID" | "HYBRID_DIESEL" | "LPG" | "OTHER" | "PETROL",
  "bodytype": "CABRIO" | "ESTATE_CAR" | "LIMOUSINE" | "OFFROAD" | "OTHER_CAR" | "SMALL_CAR" | "SPORTS_CAR" | "VAN",
  "numberOfDoors": "TWO_OR_THREE" | "FOUR_OR_FIVE" | "SIX_OR_SEVEN",
  "driveType": "ALL_WHEEL" | "FRONT" | "REAR",
  "numberOfSeats": integer,
  "performance_kw": integer,
  "cubic_capacity": integer,
  "price_max": integer,
  "mealage_max": integer,
  "first_registration_year_minimum": integer
}}

🧠 Interpret common phrases:
- “klein”, “für die Stadt”, “wenig PS” → SMALL_CAR, ≤ 1300ccm, ≤ 70 kW, FRONT
- “durchschnittlich”, “egal” → ~1600ccm, ~85kW
- “stark”, “Autobahn”, “Urlaub” → ≥2000ccm, ≥110kW

⚠️ Important:
- Convert: “1.5 Liter” → 1500ccm, “150 PS” → 110kW
- Accept: “ab 2016” → first_registration_year_minimum = 2017
- Accept: “bis 120.000 km” → mealage_max = 120000

💬 Return only the JSON. No extra explanation or comments. All strings in double quotes.
Always respond in the same language the user used in their last message.'''
    
    else:
        return fr'''You are a warm, helpful and intuitive assistant helping a person find a used car that fits their needs.

You already received some preferences in JSON format, but a few values are still missing.

🎯 Your task: Based on the user's last message, either:
→ return a new valid JSON object (if the user provides clear preferences),  
→ or respond in natural language (if the user seems confused or needs help).

❗ Only ask about parameters that exist in the following JSON schema:

{{
  "gearbox": "AUTOMATIC" | "MANUAL" | "SEMI_AUTOMATIC",
  "fueltype": "CNG" | "DIESEL" | "ELECTRICITY" | "ETHANOL" | "HYBRID" | "HYBRID_DIESEL" | "LPG" | "OTHER" | "PETROL",
  "bodytype": "CABRIO" | "ESTATE_CAR" | "LIMOUSINE" | "OFFROAD" | "OTHER_CAR" | "SMALL_CAR" | "SPORTS_CAR" | "VAN",
  "numberOfDoors": "TWO_OR_THREE" | "FOUR_OR_FIVE" | "SIX_OR_SEVEN",
  "driveType": "ALL_WHEEL" | "FRONT" | "REAR",
  "numberOfSeats": integer,
  "performance_kw": integer,
  "cubic_capacity": integer,
  "price_max": integer,
  "mealage_max": integer,
  "first_registration_year_minimum": integer
}}

Only ask about one missing value at a time. Always start with the most important one (e.g. fueltype > performance_kw). This keeps the conversation simple and natural.

❌ Do not ask about brands, models, colors, readiness, price negotiation, or any features not in the schema.

---

📌 The user’s last message was: "{last_user_message}"

🧠 Decision logic:

If the user is clear and confident (e.g. “I prefer automatic”, “max 20.000 km”, “at least 5 seats”)  
✅ then return a JSON object that adds or updates the missing values. Use only the allowed values.

If the user says “any”, “egal”, “whatever”, or “doesn’t matter” →  
✅ For categorical fields, return "any" as the value.


If the user sounds confused or unsure (e.g. says “I don’t know”, “hilf mir”, “keine Ahnung”, “what would you suggest?”)  
✅ then do NOT return JSON.  
✅ Instead, explain briefly (1–2 friendly sentences) what the missing value means,  
✅ and ask a simple follow-up question or suggest a common default.

🛑 Never mix both formats in one response.  
🛑 Never mention the word JSON or technical terms.  
✅ Be warm, simple and speak in the user’s language.'''


def extract_missing_fields(prefs):
    required_fields = ["gearbox", "fueltype", "bodytype", "numberOfDoors", "driveType",
                       "numberOfSeats", "performance_kw", "cubic_capacity", "price_max",
                       "mealage_max", "first_registration_year_minimum"]
    return [field for field in required_fields if prefs.get(field) is None]

CATEGORICAL_FIELDS = {
    "gearbox",
    "fueltype",
    "bodytype",
    "numberOfDoors",
    "driveType"
}

def call_gpt(user_input, system_prompt, temperature=0.4, max_tokens=300):

    url_new= f"{endpoint}openai/deployments/{deployment_name}/chat/completions?api-version={api_version}"
    # Prepare the headers
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }
     # Declare empty messages list
    messages = []

    # Add the system prompt
    messages.append({"role": "system", "content": system_prompt})
    
    
    for msg in st.session_state.chat_history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": user_input})
   

    # Prepare the request body
    body = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    try:
        response = requests.post(url_new, headers=headers, json=body)
        response.raise_for_status()
        data = response.json()

        return data["choices"][0]["message"]["content"]
    
    except requests.exceptions.HTTPError as e:
        return f"[HTTP ERROR] {str(e)}"
    except KeyError:
        return "[ERROR] Unexpected response format"
    except Exception as e:
        return f"[ERROR] {str(e)}" 

# Streamlit UI
st.title("💬 Azure GPT Chat")

with st.form(key="chat_form", clear_on_submit=True):

    user_input = st.text_input("You:", key="input")
    submitted = st.form_submit_button("Send")


if submitted and user_input:

    st.session_state.chat_history.append({"role": "user", "content": user_input})
    user_input_lower = user_input.lower()

    if not st.session_state.awaiting_followup:
        system_prompt = get_system_prompt("initial")
        json_text = call_gpt(user_input, system_prompt).strip()
        st.write(f"🤖 Assistant: {json_text}")

        st.session_state.current_preferences = None


        try:
            if re.match(r"^\s*\{[\s\S]*\}\s*$", json_text):
               parsed = handle_any_statements(json_text, user_input)
               if parsed:
                    # User said "I don't care about the rest"
                    if user_says_rest_doesnt_matter(user_input):
                         for field in CATEGORICAL_FIELDS:
                            if parsed.get(field) is None:
                                parsed[field] = "any"
                    st.session_state.current_preferences = parsed
               else:
                    st.session_state.current_preferences = None
            else:
                st.session_state.chat_history.append({"role": "assistant", "content": "[⚠️ Antwort war kein reines JSON – bitte noch einmal formulieren.]"})
        except Exception:
                render_chat_history()
                st.stop()
            

        if st.session_state.current_preferences is None:
            st.session_state.chat_history.append({"role": "assistant", "content": "[ERROR: Unable to parse preferences]"})
        else:
            missing_fields = extract_missing_fields(st.session_state.current_preferences)

            if missing_fields:
                follow_up_prompt = build_follow_up_prompt(
                    st.session_state.current_preferences,
                    missing_fields,
                    user_input
                )
                follow_up_question = call_gpt(
                    follow_up_prompt,
                    get_system_prompt("followup", user_input),
                    0.4,
                    150
                )
                st.session_state.chat_history.append({"role": "assistant", "content": follow_up_question})
                st.session_state.awaiting_followup = True
            else:
                st.session_state.awaiting_followup = False
                st.session_state.chat_history.append({"role": "assistant", "content": json_text})  

        render_chat_history()      

    else:
        system_prompt = get_system_prompt("initial")
        gpt_response = call_gpt(user_input, system_prompt).strip()

        followup_prefs = None
        gpt_gave_json = False

        try:
            if re.match(r"^\s*\{[\s\S]*\}\s*$", gpt_response):
                parsed = handle_any_statements(gpt_response, user_input, st.session_state.current_preferences)
                if parsed:
                    followup_prefs = parsed
                    gpt_gave_json = True
            else:
                st.session_state.chat_history.append({"role": "assistant", "content": "[⚠️ Antwort war kein reines JSON – bitte noch einmal formulieren.]"})
        except Exception:
                render_chat_history()
                st.stop()


        if not gpt_gave_json:
            st.session_state.chat_history.append({"role": "assistant", "content": gpt_response})
            render_chat_history()
            st.stop()
            
        elif followup_prefs:
        
            st.session_state.current_preferences.update({k: v for k, v in followup_prefs.items() if v is not None})
            still_missing = extract_missing_fields(st.session_state.current_preferences)

            if still_missing:
                follow_up_prompt = build_follow_up_prompt(
                    st.session_state.current_preferences,
                    still_missing,
                    user_input
                )
                follow_up_question = call_gpt(
                    follow_up_prompt,
                    get_system_prompt("followup"),
                    0.4,
                    200
                )
                st.session_state.chat_history.append({"role": "assistant", "content": follow_up_question})
                render_chat_history()
            else:
                st.session_state.awaiting_followup = False
                st.write(st.session_state.current_preferences)
                ordered_keys = [
                   "gearbox",
                   "fueltype",
                   "bodytype",
                   "numberOfDoors",
                   "driveType",
                   "numberOfSeats",
                   "performance_kw",
                   "cubic_capacity",
                   "price_max",
                   "mealage_max",
                   "first_registration_year_minimum"
                ]
                ordered_values = [st.session_state.current_preferences.get(key) for key in ordered_keys]
                st.write(ordered_values)
                query_vector = preprocess_input(
                     ordered_values[2],  # category
                     ordered_values[3],  # doors
                     ordered_values[10],  # first_reg
                     ordered_values[0],  # gearbox
                     str(ordered_values[5]),  # seats
                     ordered_values[1],  # fuel_type
                     ordered_values[6],  # performance
                     ordered_values[4],  # drivetype
                     ordered_values[7]   # cubiccapacity
                )
              
                results, count_results = search_similar_cars_without_filters(
                        query_vector,
                        numberofcars,
                        similarity_threshold=percentagefinal,
                )
                if results:
                 # Filtering the data depends on the choice of the user
                 st.markdown("<br>", unsafe_allow_html=True)
                 for car in results:
            
                     car_data = car["_source"]       
                     real_ID = car_data["CarID"]
                     full_car_info = get_car_by_id(real_ID)
        
                     if full_car_info:
                         st.write(f"🆔 ID: {full_car_info['CarID']}")
                         st.write(f"🔥 Body Type: {full_car_info.get('BodyType', 'N/A')}")
                         st.write(f"📏 Make: {full_car_info['Make']}  | 📏 Model: {full_car_info.get('Model', 'N/A')} ")
                         st.write(f"⚙️ Gearbox: {full_car_info.get('GearBox', 'N/A')} | ⛽ Fuel Type : {full_car_info.get('Fuel', 'N/A')}")
                         st.write(f"💡 Body Color: {full_car_info.get('BodyColor', 'N/A')} | 🚪 Doors : {full_car_info.get('NumberOfDoors', 'N/A')}")
                         st.write(f"🚙 Drive Type: {full_car_info.get('DriveType', 'N/A')} | 🚗📏 Mileage : {full_car_info.get('Mileage', 'N/A')}")
                         st.write(f"🏁 Cubic Capacity: {full_car_info.get('CubicCapacity', 'N/A')} | ⚡ Performance : {full_car_info.get('Power', 'N/A')}")
                         st.write(f"👥 Number Of Seats: {full_car_info.get('NumberOfSeats', 'N/A')} | 🛠️ Usage State : {full_car_info.get('UsageState', 'N/A')}")
                         st.write(f"📅 First Registration: {full_car_info.get('FirstRegistration', 'N/A')} | 💰 Price: {full_car_info.get('Price', 'N/A')}")
                         #st.write(f"📅 Score : {car['_score']}")
                         st.write("---")
                     else:
                      st.write(f"❌ Car with ID {real_ID} not found in DynamoDB.")
                else:
                    st.write("❌ No similar cars found.")



