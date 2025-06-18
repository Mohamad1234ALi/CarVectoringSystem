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

    if gearbox_needed :  # checkbox True and value provided
        filters.append({"term": {"GearBox": gearbox_value}})
    # Fuel type filter
    if fuel_needed :  # checkbox True and value provided
        filters.append({"term": {"Fuel": fuel_value}})

    if category_needed :  # checkbox True and value provided
        filters.append({"term": {"BodyType": category_value}})

    if doors_needed :  # checkbox True and value provided
        filters.append({"term": {"NumberOfDoors": doors_value}})

    if drive_needed :  # checkbox True and value provided
        filters.append({"term": {"DriveType": drive_value}})

    if seats_needed :  # checkbox True and value provided
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

    if gearbox_needed :  # checkbox True and value provided
        filters.append({"term": {"GearBox": gearbox_value}})

    # Fuel type filter
    if fuel_needed :  # checkbox True and value provided
        filters.append({"term": {"Fuel": fuel_value}})

    if category_needed :  # checkbox True and value provided
        filters.append({"term": {"BodyType": category_value}})

    if doors_needed :  # checkbox True and value provided
        filters.append({"term": {"NumberOfDoors": doors_value}})

    if drive_needed :  # checkbox True and value provided
        filters.append({"term": {"DriveType": drive_value}})

    if seats_needed :  # checkbox True and value provided
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
   gearbox = st.selectbox("Gearbox", categories_list[CATEGORICAL_FEATURES.index("GearBox")])
    
with col2:
   st.markdown("######")
   gearbox_needed = st.checkbox("Filter by Gearbox", value=False)

col1, col2 = st.columns(2)

with col1:
    fuel_type = st.selectbox("Fuel Type", categories_list[CATEGORICAL_FEATURES.index("Fuel")])
    
with col2:
    st.markdown("######")
    fuel_needed = st.checkbox("Filter by Fuel",  value=False)

col1, col2 = st.columns(2)

with col1:
    category = st.selectbox("Body Type", categories_list[CATEGORICAL_FEATURES.index("BodyType")])
    
with col2:
    st.markdown("######")
    category_needed = st.checkbox("Filter by Body Type",  value=False)

col1, col2 = st.columns(2)

with col1:
    doors = st.selectbox("Number Of Doors", categories_list[CATEGORICAL_FEATURES.index("NumberOfDoors")])
    
with col2:
    st.markdown("######")
    doors_needed = st.checkbox("Filter by Doors",  value=False)

col1, col2 = st.columns(2)

with col1:
    drivetype = st.selectbox("Drive Type", categories_list[CATEGORICAL_FEATURES.index("DriveType")])
    
with col2:
    st.markdown("######")
    drive_needed = st.checkbox("Filter by Drive Type",  value=False)

col1, col2 = st.columns(2)

with col1:
    seats = st.selectbox("Number Of Seats", categories_list[CATEGORICAL_FEATURES.index("NumberOfSeats")])
    
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


currentPreferences = {} 
url = f"{endpoint}openai/deployments/{deployment_name}/chat/completions?api-version={api_version}"
headers = {
    "Content-Type": "application/json",
    "api-key": api_key
}

def merge_preferences(current: dict, updates: dict) -> None:
    if updates.get("gearbox"): current["gearbox"] = updates["gearbox"]
    if updates.get("fueltype"): current["fueltype"] = updates["fueltype"]
    if updates.get("bodytype"): current["bodytype"] = updates["bodytype"]
    if updates.get("numberOfDoors"): current["numberOfDoors"] = updates["numberOfDoors"]
    if updates.get("driveType"): current["driveType"] = updates["driveType"]

    if updates.get("numberOfSeats") is not None: current["numberOfSeats"] = updates["numberOfSeats"]
    if updates.get("performance_kw") is not None: current["performance_kw"] = updates["performance_kw"]
    if updates.get("cubic_capacity") is not None: current["cubic_capacity"] = updates["cubic_capacity"]
    if updates.get("price_max") is not None: current["price_max"] = updates["price_max"]
    if updates.get("mealage_max") is not None: current["mealage_max"] = updates["mealage_max"]
    if updates.get("first_registration_year_minimum") is not None:
        current["first_registration_year_minimum"] = updates["first_registration_year_minimum"]





awaitingFollowUp = False
if "messages" not in st.session_state:
    st.session_state.messages = []


car_initial_prompt = (
    "You are a smart and helpful car assistant. Your task is to understand what kind of car the user is looking for, based on natural language.\n\n"
    "You will extract their wishes and return them as a valid JSON object using the format and allowed values below.\n\n"
    "🎯 Your goal is to fill all fields that can be reasonably inferred. Use the exact terms shown. If something is unclear or missing, set it to `null`.\n\n"
    "Allowed values and structure:\n\n"
    "{\n"
    '  "gearbox": "AUTOMATIC" | "MANUAL" | "SEMI_AUTOMATIC",\n'
    '  "fueltype": "CNG" | "DIESEL" | "ELECTRICITY" | "ETHANOL" | "HYBRID" | "HYBRID_DIESEL" | "LPG" | "OTHER" | "PETROL",\n'
    '  "bodytype": "CABRIO" | "ESTATE_CAR" | "LIMOUSINE" | "OFFROAD" | "OTHER_CAR" | "SMALL_CAR" | "SPORTS_CAR" | "VAN",\n'
    '  "numberOfDoors": "TWO_OR_THREE" | "FOUR_OR_FIVE" | "SIX_OR_SEVEN",\n'
    '  "driveType": "ALL_WHEEL" | "FRONT" | "REAR",\n'
    '  "numberOfSeats": integer,\n'
    '  "performance_kw": integer,\n'
    '  "cubic_capacity": integer,\n'
    '  "price_max": integer,\n'
    '  "mealage_max": integer,\n'
    '  "first_registration_year_minimum": integer,\n'
    "}\n\n"
    '- “klein”, “für die Stadt”, “wenig PS” → SMALL_CAR, ≤ 1300ccm, ≤ 70 kW, FRONT\n'
    '- “durchschnittlich”, “egal” → ~1600ccm, ~85kW\n'
    '- “stark”, “Autobahn”, “Urlaub” → ≥2000ccm, ≥110kW\n\n'
    "⚠️ Important:\n"
    '- Convert: “1.5 Liter” → 1500ccm, “150 PS” → 110kW\n'
    '- Accept: “ab 2016” → `first_registration_year_minimum = 2017`\n'
    '- Accept: “bis 120.000 km” → `mealage_max = 120000`\n\n'
    '💬 Return **only** the JSON. No extra explanation or comments. All strings in double quotes.'
)


followup_instruction = """
You are a warm, helpful and intuitive assistant helping a person find a used car that fits their needs.

You already received some preferences in JSON format, but a few values are still missing.

🎯 Your task: Ask ONE friendly, natural follow-up question to clarify one or more of the missing parameters.

❗ Only ask about parameters that exist in the following JSON schema:

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

❌ Do not ask about brands, models, colors, readiness, price negotiation, or any features not in the schema.

---

📌 The user’s last message was: "{lastUserMessage}"

If the user sounds confused or unsure (e.g. says “I don’t know”, “hilf mir”, “keine Ahnung”),  
✅ then explain briefly what the missing value means, and ask in simpler terms.

If the user responds with a question like “Which is better?”, “What would you suggest?”, “Welcher Preis ist gut?”,  
✅ then offer a reasonable example value (based on common sense or previous answers),  
✅ and follow it with a kind question like “Would that work for you?”

---

🛑 Never repeat the same sentence twice.  
🛑 Do not mention JSON or technical terms.  
✅ Always use one friendly sentence, in the user’s language (usually German).
""".strip()



def get_system_prompt(phase, last_user_message=""):
    if phase == "initial":
        return car_initial_prompt
    elif phase == "followup":
        template = followup_instruction  # or any constant string you use
        return template.replace("{lastUserMessage}", last_user_message)
    else:
        return "You are an AI, a helpful assistant"

def render_chat_history():
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"**🧑 You:** {msg['content']}")
        elif msg["role"] == "assistant":
            st.markdown(f"**🤖 Assistant:** {msg['content']}")


def get_gpt_message(user_input, system_prompt, temperature_value, max_tokens):
    # Prepare the headers
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }

    # Build the full message list
    messages = [{"role": "system", "content": system_prompt}]
    
    for msg in st.session_state.messages:
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    messages.append({"role": "user", "content": user_input})

    # Prepare the request body
    body = {
        "messages": messages,
        "temperature": temperature_value,
        "max_tokens": max_tokens
    }

    try:
        response = requests.post(url, headers=headers, json=body)
        response.raise_for_status()
        data = response.json()

        return data["choices"][0]["message"]["content"]
    
    except requests.exceptions.HTTPError as e:
        return f"[HTTP ERROR] {str(e)}"
    except KeyError:
        return "[ERROR] Unexpected response format"
    except Exception as e:
        return f"[ERROR] {str(e)}"


def build_followup_prompt(prefs, missing_fields, language="en", last_user_message=""):
    # Serialize preferences as pretty JSON
    prefs_json = json.dumps(prefs, indent=2)

    # Language hint mapping
    lang_hint = {
        "de": "German",
        "fr": "French",
        "it": "Italian",
        "ar": "Arabic"
    }.get(language, "English")

    # Build the full prompt
    prompt = f"""
You are helping a user find a suitable used car.

Based on the previous message, we extracted the following preferences:

{prefs_json}

Some values are still missing: {", ".join(missing_fields)}.

The user just wrote: "{last_user_message}".

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

Return only the follow-up question – in {lang_hint}.
""".strip()

    return prompt

# Streamlit UI
st.title("💬 Azure GPT Chat")

with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("You:", key="input")
    submitted = st.form_submit_button("Send")

if submitted and user_input:

     # Append user message to memory
    st.session_state.messages.append({"role": "user", "content": user_input})


    if not awaitingFollowUp:
        response = get_gpt_message(user_input, get_system_prompt("initial"), 0.4, 150)
        try:
            currentPreferences = json.loads(response)
            null_fields = [key for key, value in currentPreferences.items() if value is None]

            if null_fields:
               
               followUpPrompt = build_followup_prompt(currentPreferences, null_fields, "en", last_user_message=user_input)  
               followUpQuestion = get_gpt_message(followUpPrompt, get_system_prompt("followup",user_input), 0.4, 150); 
               st.session_state.messages.append({"role": "assistant", "content": followUpQuestion})
               st.write("not followupwaiting but nll")
               awaitingFollowUp = True
       
            
            else:
               awaitingFollowUp = False
               st.session_state.messages.append({"role": "assistant", "content": response})
               st.write("not followupwaiting but finish")
        
               
            
            render_chat_history()
        except json.JSONDecodeError:
         st.warning("The response is not valid JSON:")
         st.write(response)

    else:   
        # If we are awaiting a follow-up, just show the last question
        followupresponse = get_gpt_message(user_input, get_system_prompt("followup"), 0.4, 300)

        try:

            followupPrefs = json.loads(followupresponse)
            merge_preferences(currentPreferences, followupPrefs)
            still_null_fields = [key for key, value in currentPreferences.items() if value is None]

            confused_keywords = ["hilfe", "hilf", "weiß nicht", "keine ahnung", "help"]
            user_is_confused = any(keyword in user_input.lower() for keyword in confused_keywords)

            if user_is_confused:
               help = build_followup_prompt(currentPreferences, still_null_fields, "en", last_user_message=user_input)  
               helpQuestion = get_gpt_message(help, get_system_prompt("followup"), 0.4, 150); 
               st.session_state.messages.append({"role": "assistant", "content": helpQuestion})
               st.write("here open is confused")
               render_chat_history()
              

            if still_null_fields:
               
               folowhelp = build_followup_prompt(currentPreferences, still_null_fields, "en", last_user_message=user_input)  
               followqt = get_gpt_message(folowhelp, get_system_prompt("followup"), 0.4, 150); 
               st.session_state.messages.append({"role": "assistant", "content": followqt})
                     
               st.write("here open is still null")
               render_chat_history()
            else:
               awaitingFollowUp = False
               st.write("here open is not null")
               # st.session_state.messages.append({"role": "assistant", "content": final_message})
               # render_chat_history()
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
               ordered_values = [currentPreferences.get(key) for key in ordered_keys]
               st.write(ordered_values)
               query_vector = preprocess_input(
                     ordered_values[2],  # category
                     ordered_values[3],  # doors
                     ordered_values[10],  # first_reg
                     ordered_values[0],  # gearbox
                     ordered_values[5],  # seats
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

        except json.JSONDecodeError:
         st.warning("The response is not valid JSON:")
         st.write(followupresponse)           


