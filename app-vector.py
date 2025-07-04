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
import openai
from openai import AzureOpenAI

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


client_azure = AzureOpenAI(
    api_key=api_key,
    api_version="2024-12-01-preview",
    azure_endpoint=endpoint
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
  
    if seats == -1:
        seats = "Any"

    if str(seats) == "-1":
        seats = "Any"    


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
        0 if first_reg in ["Any", None, -1] else first_reg,
        0 if performance in ["Any", None, -1] else performance,
        0 if cubiccapacity in ["Any", None, -1] else cubiccapacity
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
  

def get_embedding(text):
    response = client_azure.embeddings.create(
        input=text,
        model="text-embedding-3-large"  # Azure deployment name (not model name)
    )
    return response.data[0].embedding

def search_similar_cars_without_filters(
    user_inputs, 
    numberofcars, 
    similarity_threshold,
    
):
  
    # Construct the query with bool filter and knn must
    description = generate_description(user_inputs)
    query_vector = get_embedding(description)

    query = {
        "size": numberofcars * 10,
        "query": {
            "knn": {
                "description_vector": {  # Use the field name for description vectors
                    "vector": query_vector,
                    "k": numberofcars * 10
                }
            }
        }
    }

    response = client.search(index=INDEX_NAME, body=query)
    results = response["hits"]["hits"]

    count_result = len(results)

    # Optional: filter results by similarity threshold on _score
    filtered = [r for r in results if r["_score"] >= similarity_threshold]
    random.shuffle(filtered)
    return filtered[:numberofcars], count_result

def test(
    user_inputs, 
    numberofcars, 
    similarity_threshold,
    
):
  
    # Construct the query with bool filter and knn must
    description = generate_description(user_inputs)
    query_vector = get_embedding(description)

    return description, query_vector




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
    

def generate_description(user_inputs):
    return (
        f"This car is a {user_inputs['bodytype']} with {user_inputs['driveType']} drive, "
        f"{user_inputs['gearbox']} gearbox, runs on {user_inputs['fueltype']}, has {user_inputs['numberOfDoors']} doors and "
        f"{user_inputs['numberOfSeats']} seats. It was first registered in {user_inputs['first_registration_year_minimum']}, "
        f"delivers {user_inputs['performance_kw']} kW of power and has a cubic capacity of {user_inputs['cubic_capacity']} cc. "
        f"It costs {user_inputs['price_max']} euros and has {user_inputs['mealage_max']} km mileage."
    )


# Build follow-up prompt dynamically from preferences and missing fields
def build_follow_up_prompt(prefs, missing_fields, last_user_message=""):
    #json_formatted = json.dumps(prefs, indent=2)
   

    json_str = json.dumps(prefs, indent=4, ensure_ascii=False)

    prompt = f'''
You are helping a user find a suitable used car.

Based on the previous message, we extracted the following preferences:

{json_str}

Some values are still missing: {", ".join(missing_fields)}.

The user just wrote: "{last_user_message}".

🎯 Your task:
→ If the user clearly gives one of the missing values, return a new valid JSON object with only that update.
→ If the user sounds unsure or confused, respond in natural language. Do NOT return JSON in that case.

❗ If the user says “egal”, “any” or “doesn’t matter” for a missing field, set only that specific field to "any".
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
    '''

    return prompt.strip()

# System prompts

def get_system_prompt(phase, last_user_message=""):
    if phase == "initial":
        return """
You are a smart and helpful car assistant. Your task is to understand what kind of car the user is looking for, based on natural language.

You will extract their preferences and return them as a valid JSON object using the format and allowed values below.

🎯 Your goal is to fill all fields that can be reasonably inferred from the user's input. Use the exact terms shown.  
If a field is missing, unclear, or cannot be inferred, set it to null.

❗ If the user explicitly says they do NOT want something (e.g. "no limousine", "not electric", "not diesel"),  
✅ then set the corresponding value to null unless a clear preferred alternative is provided.  

🟢 🟢 If the user expresses indifference (e.g., “egal”, “beliebig”, “keine Präferenz”) for a specific field, then and only then set that field to "any" (for text fields) or -1 (for numeric fields).

🔒 Do not assume that the user is indifferent to all other fields.

🛑 Never set fields to "any" or -1 **unless** the user has explicitly expressed indifference for that **specific** field.

👂 Wait until the user has answered a question about a specific field. If the user says "egal" or similar, then apply the rule — otherwise leave the field as null or skip it entirely.

🔁 If the user provides approximate values (e.g. “ca. 90 PS”), convert to a rounded integer in kW or ccm where possible.

❓ If the user's statements are contradictory, unclear, or invalid (e.g. unknown fuel types), set the affected field(s) to null.

---

Allowed values and expected format:

{
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
}

---

🧠 Interpret common phrases and convert accordingly:

- “klein”, “für die Stadt”, “wenig PS” → "bodytype": "SMALL_CAR", "cubic_capacity" ≤ 1300, "performance_kw" ≤ 70, "driveType": "FRONT"
- “durchschnittlich”, “normal”, “egal” (in context of engine) → "cubic_capacity" ≈ 1600, "performance_kw" ≈ 85
- “stark”, “Autobahn”, “Urlaub”, “schnell”, “kräftig” → "cubic_capacity" ≥ 2000, "performance_kw" ≥ 110
- “1.5 Liter” → "cubic_capacity": 1500
- “150 PS” → "performance_kw": 110
- “ab 2016” → "first_registration_year_minimum": 2017
- “bis 120.000 km” → "mealage_max": 120000
- “7-Sitzer”, “für 7 Personen” → "numberOfSeats": 7
- “5 Personen”, “für die Familie” → "numberOfSeats": 5

---

💬 The response must be a valid JSON object only – no text, no comments, no explanation.  
🚫 Do not write anything except the JSON object.  
Always respond in the same language the user used in their last message.
🔚 If all fields in the JSON object are filled with specific values — that is:
- no field is set to "any" (for categorical fields),
- no field is set to -1 (for numerical fields),
- and no field is null —

→ Then immediately stop asking further questions  
→ and return the final JSON object with all collected values.

✅ Do not wait for the user to say "done" or "show me the cars".
""".strip()

    elif phase == "followup":
        template = """
You are a warm, helpful and intuitive assistant helping a person find a used car that fits their needs.

You already received some preferences in JSON format, but a few values are still missing.

🎯 Your task: Based on the user's last message, either:  
→ return a new valid JSON object (if the user provides clear preferences),  
→ or respond in natural language (if the user seems unsure or needs help).

If a field is missing, unclear, or cannot be inferred, set it to null.  
🟢 If the user expresses indifference (e.g., “egal”, “beliebig”, “keine Präferenz”) for a specific field, then and only then set that field to "any" (for text fields) or -1 (for numeric fields).

🔒 Do not assume that the user is indifferent to all other fields.

🛑 Never set fields to "any" or -1 **unless** the user has explicitly expressed indifference for that **specific** field.

👂 Wait until the user has answered a question about a specific field. If the user says "egal" or similar, then apply the rule — otherwise leave the field as null or skip it entirely.❗ Only ask about parameters defined in the following JSON schema:

{
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
}

🧠 Interpret common phrases:
- “klein”, “für die Stadt”, “wenig PS” → SMALL_CAR, ≤ 1300ccm, ≤ 70 kW, FRONT
- “durchschnittlich”, “egal” (motor-related) → ~1600ccm, ~85kW
- “stark”, “Autobahn”, “Urlaub”, “kräftig” → ≥2000ccm, ≥110kW
- “1.5 Liter” → 1500ccm, “150 PS” → 110kW
- “ab 2016” → first_registration_year_minimum = 2017
- “bis 120.000 km” → mealage_max = 120000
- “5 Personen”, “Familienauto” → numberOfSeats = 5
- “7-Sitzer”, “für 7 Leute” → numberOfSeats = 7

Only ask about one missing field at a time. Always begin with the most relevant one (e.g. fueltype > performance_kw).

💬 If the user gives a clear answer (e.g. “I prefer automatic”, “max 20.000 km”, “at least 5 seats”)  
✅ return a new full JSON object, updated with this information. Use only the allowed values.

🤝 If the user seems unsure (e.g. says “keine Ahnung”, “was meinst du?”, “hilf mir”, “I don’t know”)  
✅ do not return JSON. Instead:  
– explain in 1–2 friendly sentences what the missing value means  
– ask a helpful, simple follow-up question or suggest a common option

🛑 Never mix JSON and natural language in one response.  
🛑 Never mention the word “JSON”, “schema”, “format” or any technical terms.  
✅ Always reply in the same language the user used.
✅ Be warm, respectful and easy to understand.
🔚 If all fields in the JSON object are filled with specific values — that is:
- no field is set to "any" (for categorical fields),
- no field is set to -1 (for numerical fields),
- and no field is null —

→ Then immediately stop asking further questions
→ and return the final JSON object with all collected values.

✅ Do not wait for the user to say "done" or "show me the cars".

""".strip()
        return template

    else:
        raise ValueError(f"Unknown prompt phase: {phase}")

def extract_json_from_response(response: str):
    match = re.search(r'\{[\s\S]*\}', response)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError as e:
            print("❌ JSON decode failed:", e)
            return None
    return None

def extract_missing_fields(prefs):
    required_fields = ["gearbox", "fueltype", "bodytype", "numberOfDoors", "driveType",
                       "numberOfSeats", "performance_kw", "cubic_capacity", "price_max",
                       "mealage_max", "first_registration_year_minimum"]
    return [field for field in required_fields if prefs.get(field) is None]


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


        st.session_state.current_preferences = None

    
        try:
            if re.match(r"^\s*\{[\s\S]*\}\s*$", json_text):
                st.session_state.current_preferences = json.loads(json_text)
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
                st.write("still missing part and true")
                st.session_state.chat_history.append({"role": "assistant", "content": follow_up_question})
                st.session_state.awaiting_followup = True
            else:
                st.session_state.awaiting_followup = False
                st.session_state.chat_history.append({"role": "assistant", "content": json_text})  

        render_chat_history()      

    else:
        system_prompt = get_system_prompt("followup", user_input)
        gpt_response = call_gpt(user_input, system_prompt, 0.4, 300).strip()

        followup_prefs = None
        gpt_gave_json = False

        try:
            if re.match(r"^\s*\{[\s\S]*\}\s*$", gpt_response):
                followup_prefs = json.loads(gpt_response)
                gpt_gave_json = True
            else:
                st.session_state.chat_history.append({"role": "assistant", "content": "[⚠️ Antwort war kein reines JSON – bitte noch einmal formulieren.]"})
        except Exception:
                render_chat_history()
                st.stop()


        if not gpt_gave_json:
            st.write("GPT gave no JSON")
            st.session_state.chat_history.append({"role": "assistant", "content": gpt_response})
            parsed_json = extract_json_from_response(gpt_response)
            st.write(gpt_response)
            if parsed_json:
                still_missing_check = extract_missing_fields(parsed_json)
                if still_missing_check:
                    st.write(" missing fields found. ha")
                    render_chat_history()
                    st.stop()
                else:
                    st.write("No missing fields found. finished")
                    st.session_state.awaiting_followup = False
                    st.write(parsed_json)
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
                   
                    ordered_values = [parsed_json.get(key) for key in ordered_keys]
                    user_inputs = dict(zip(ordered_keys, ordered_values)) 
                    st.write(user_inputs)   
                    # results, count_results = search_similar_cars_without_filters(
                    #     user_inputs,
                    #     numberofcars,
                    #     similarity_threshold=percentagefinal,
                    # )

                    desc, query_vec = test(
                        user_inputs,
                        numberofcars,
                        similarity_threshold=percentagefinal,
                    )

                    st.write(f"🔍 Found {desc} similar cars using cosine")
                    st.write(f"🔍 Found {query_vec}")

                    # if results:
                    #     # Filtering the data depends on the choice of the user
                    #     st.markdown("<br>", unsafe_allow_html=True)
                    #     for car in results:
            
                    #         car_data = car["_source"]       
                    #         real_ID = car_data["CarID"]
                    #         full_car_info = get_car_by_id(real_ID)
        
                    #         if full_car_info:
                    #              st.write(f"🆔 ID: {full_car_info['CarID']}")
                    #              st.write(f"🔥 Body Type: {full_car_info.get('BodyType', 'N/A')}")
                    #              st.write(f"📏 Make: {full_car_info['Make']}  | 📏 Model: {full_car_info.get('Model', 'N/A')} ")
                    #              st.write(f"⚙️ Gearbox: {full_car_info.get('GearBox', 'N/A')} | ⛽ Fuel Type : {full_car_info.get('Fuel', 'N/A')}")
                    #              st.write(f"💡 Body Color: {full_car_info.get('BodyColor', 'N/A')} | 🚪 Doors : {full_car_info.get('NumberOfDoors', 'N/A')}")
                    #              st.write(f"🚙 Drive Type: {full_car_info.get('DriveType', 'N/A')} | 🚗📏 Mileage : {full_car_info.get('Mileage', 'N/A')}")
                    #              st.write(f"🏁 Cubic Capacity: {full_car_info.get('CubicCapacity', 'N/A')} | ⚡ Performance : {full_car_info.get('Power', 'N/A')}")
                    #              st.write(f"👥 Number Of Seats: {full_car_info.get('NumberOfSeats', 'N/A')} | 🛠️ Usage State : {full_car_info.get('UsageState', 'N/A')}")
                    #              st.write(f"📅 First Registration: {full_car_info.get('FirstRegistration', 'N/A')} | 💰 Price: {full_car_info.get('Price', 'N/A')}")
                    #              #st.write(f"📅 Score : {car['_score']}")
                    #              st.write("---")
                    #         else:
                    #              st.write(f"❌ Car with ID {real_ID} not found in DynamoDB.")
                    # else:
                    #     st.write("❌ No similar cars found.")
            else:
                st.write("Failed to parse JSON from GPT response")
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
                st.write("still missing part")
                st.session_state.chat_history.append({"role": "assistant", "content": follow_up_question})
                render_chat_history()
                
            else:
                st.write("Finished collecting preferences! 🎉")
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


