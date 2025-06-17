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


# Get the parameters from the settings in streamlit app
aws_access_key = st.secrets["AWS_ACCESS_KEY_ID"]
aws_secret_key = st.secrets["AWS_SECRET_ACCESS_KEY"]
aws_region = st.secrets["AWS_DEFAULT_REGION"]

api_key = st.secrets["api_key"]
endpoint = st.secrets["endpoint"]
deployment_name = st.secrets["deployment_name"]
api_version = st.secrets["api_version"]


url = f"{endpoint}openai/deployments/{deployment_name}/chat/completions?api-version={api_version}"
headers = {
    "Content-Type": "application/json",
    "api-key": api_key
}


awaitingFollowUp = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize session state for memory
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": (
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
        }
    ]

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



def get_system_prompt(phase, last_user_message=""):
    if phase == "initial":
        return car_initial_prompt
    elif phase == "followup":
        template = last_user_message  # or any constant string you use
        return template.replace("{lastUserMessage}", last_user_message)
    else:
        return "You are an AI, a helpful assistant"




def get_gpt_message(user_input, system_prompt, temperature_value, max_tokens, chat_history, api_key, endpoint, deployment_name):
    # Prepare the headers
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }

    # Build the full message list
    messages = [{"role": "system", "content": system_prompt}]
    
    for msg in chat_history:
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

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    elif msg["role"] == "assistant":
        st.markdown(f"**Assistant:** {msg['content']}")

# Streamlit UI
st.title("💬 Azure GPT Chat")
user_input = st.text_input("You:", key="input")


if st.button("Send")  and user_input :

     # Append user message to memory
    st.session_state.messages.append({"role": "user", "content": user_input})


    if not awaitingFollowUp:
        response = get_gpt_message(user_input, get_system_prompt("initial"), 0.4, 200, st.session_state.messages, api_key, endpoint, deployment_name)
        try:
            parsed_json = json.loads(response)
            null_fields = [key for key, value in parsed_json.items() if value is None]

            if null_fields:

                print("Missing fields:", null_fields)
            else:
                print("All fields filled:", parsed_json)
   
        except json.JSONDecodeError:
         st.warning("The response is not valid JSON:")
         st.write(response)
