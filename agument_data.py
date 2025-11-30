import os
import pandas as pd
import random
import requests
import json
import time

# --- Configuration ---
SOURCE_FILE = os.path.join('data', '243_specificenergy.csv')
OUTPUT_FILE = os.path.join('data', 'augmented_training_data.csv')
NUM_EXAMPLES = 20  # Number of real data points to show the LLM
NUM_TO_GENERATE = 200  # Number of new data points to create

# These MUST match the columns used in train_model.py
FEATURES_LIST = [
    'P (MPa)',
    'mf (kg/min)',
    'v (mm/min)',
    'df (mm)',
    'do (mm)'
]
TARGET_COLUMN = 'h (mm)'
ALL_COLUMNS = FEATURES_LIST + [TARGET_COLUMN]

# --- Gemini API Configuration ---
# NOTE: Leave apiKey as "" - it is supplied automatically.
from dotenv import load_dotenv
import os

load_dotenv()  # loads .env file
API_KEY = os.getenv("API_KEY")  # reads key from environment
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={API_KEY}"


def get_few_shot_examples(df, num_examples):
    """
    Selects a random sample from the dataframe and formats it as a string.
    """
    # Ensure all columns are numeric, drop any rows that failed conversion
    df_numeric = df[ALL_COLUMNS].apply(pd.to_numeric, errors='coerce').dropna()

    if len(df_numeric) < num_examples:
        sample_df = df_numeric
    else:
        sample_df = df_numeric.sample(n=num_examples)

    # Convert to a list of dictionaries for the prompt
    examples = sample_df.to_dict('records')
    # Convert list of dicts to a JSON-like string for the prompt
    return json.dumps(examples, indent=2)


def get_response_schema():
    """
    Defines the strict JSON schema the LLM MUST output.
    """
    properties = {col: {"type": "NUMBER"} for col in ALL_COLUMNS}

    return {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": properties,
            "propertyOrdering": ALL_COLUMNS
        }
    }


def call_gemini_api(system_prompt, user_prompt, schema):
    """
    Calls the Gemini API with structured JSON output and retry logic.
    """
    print("Calling Gemini API... This may take a moment.")

    payload = {
        "contents": [{
            "parts": [{"text": user_prompt}]
        }],
        "systemInstruction": {
            "parts": [{"text": system_prompt}]
        },
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": schema
        }
    }

    headers = {'Content-Type': 'application/json'}

    max_retries = 5
    delay = 2  # start with 2 seconds

    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=300)

            if response.status_code == 200:
                print("API call successful. Parsing response...")
                return response.json()  # Success

            elif response.status_code in [429, 500, 503]:
                # Throttling or server error, wait and retry
                print(f"API Error (Status {response.status_code}). Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2
            else:
                # Other error (like 400 Bad Request - probably a prompt issue)
                print(f"API Error (Status {response.status_code}): {response.text}")
                return {"error": response.text}

        except requests.exceptions.RequestException as e:
            print(f"Network error: {e}. Retrying in {delay}s...")
            time.sleep(delay)
            delay *= 2

    print("Max retries exceeded. API call failed.")
    return {"error": "Max retries exceeded"}


def main():
    """
    Main function to run the data augmentation.
    """
    print(f"Loading original data from {SOURCE_FILE}...")
    try:
        df_original = pd.read_csv(SOURCE_FILE)
    except FileNotFoundError:
        print(f"FATAL ERROR: Source file not found at {SOURCE_FILE}")
        print("Please make sure your 243_specificenergy.csv file is in the 'data' folder.")
        return

    # 1. Get examples
    example_string = get_few_shot_examples(df_original, NUM_EXAMPLES)

    # 2. Define schema
    schema = get_response_schema()

    # 3. Build prompts
    system_prompt = (
        "You are a physics expert and data scientist specializing in Abrasive Waterjet machining. "
        "Your task is to generate new, physically realistic data points based on the examples provided. "
        "The relationships between inputs (P, mf, v, df, do) and the output (h) are complex and non-linear. "
        "Ensure the generated data follows these physical trends:"
        " - Higher 'P (MPa)' or 'mf (kg/min)' generally increases 'h (mm)'."
        " - Higher 'v (mm/min)', 'df (mm)', or 'do (mm)' generally decreases 'h (mm)'."
        "You must only output a valid JSON array matching the provided schema."
    )

    user_prompt = (
        f"Here are {NUM_EXAMPLES} real-world examples of data:\n"
        f"{example_string}\n\n"
        f"Please generate {NUM_TO_GENERATE} new, unique, and physically realistic data points that follow the same "
        f"patterns and statistical distributions. Adhere strictly to the JSON schema."
    )

    # 4. Call API
    response_json = call_gemini_api(system_prompt, user_prompt, schema)

    if "error" in response_json:
        print(f"Failed to generate data: {response_json['error']}")
        return

    # 5. Parse response
    try:
        # Extract the text content, which is a JSON string
        text_data = response_json['candidates'][0]['content']['parts'][0]['text']
        # Parse that string into a Python list of dictionaries
        generated_data_list = json.loads(text_data)

        if not isinstance(generated_data_list, list):
            print("Error: LLM did not return a JSON array.")
            return

        print(f"Successfully generated {len(generated_data_list)} new data points.")

    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as e:
        print(f"Error parsing LLM response: {e}")
        print("Response dump:", response_json)
        return

    # 6. Combine and Save
    try:
        new_data_df = pd.DataFrame(generated_data_list)

        # Ensure new data has correct columns
        new_data_df = new_data_df[ALL_COLUMNS]

        # Combine old and new data
        print(f"Original data size: {len(df_original)} rows")
        print(f"New data size: {len(new_data_df)} rows")

        # Use only the essential columns from the original data
        df_original_clean = df_original[ALL_COLUMNS].apply(pd.to_numeric, errors='coerce').dropna()

        combined_df = pd.concat([df_original_clean, new_data_df], ignore_index=True)

        print(f"Combined data size: {len(combined_df)} rows")

        # Save to new file
        os.makedirs('data', exist_ok=True)
        combined_df.to_csv(OUTPUT_FILE, index=False)

        print(f"\n--- Data Augmentation Successful ---")
        print(f"New dataset saved to: {OUTPUT_FILE}")
        print(f"You can now run 'train_model.py' to train on this {len(combined_df)}-row dataset.")

    except KeyError:
        print("Error: The generated data is missing one or more required columns.")
        print("Required:", ALL_COLUMNS)
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")


if __name__ == "__main__":
    main()

