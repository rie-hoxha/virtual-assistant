import logging
from flask import Flask, request, jsonify
from huggingface_hub import InferenceClient
import uuid
import pandas as pd
import numpy as np
from google.oauth2 import service_account
from google.cloud import dialogflow_v2 as dialogflow
from sentence_transformers import SentenceTransformer
import re
import os
from dotenv import load_dotenv
from numpy.linalg import norm

# Setup logging configuration
logging.basicConfig(level=logging.DEBUG)

# Initialize the Flask application
app = Flask(__name__)

# Load environment variables from the .env file
load_dotenv()

# Load Dialogflow credentials
logging.debug("Loading Dialogflow credentials")
credentials = service_account.Credentials.from_service_account_file('chat-bot-test-417314-0db7f8c4b1a4.json')
project_id = 'chat-bot-test-417314'  # Replace with your actual project ID
session_client = dialogflow.SessionsClient(credentials=credentials)

# Load the dataframe with smartphone data
logging.debug("Loading smartphone data")
data = pd.read_csv('flipkart_smartphones.csv')

# Check if the data is loaded correctly
if data.empty:
    logging.error("Dataframe is empty. Please check the CSV file.")
else:
    logging.debug(f"Dataframe shape: {data.shape}")

# Load the precomputed embeddings
logging.debug("Loading precomputed embeddings")
embeddings = np.load('smartphone_embeddings.npy')
logging.debug(f"Embeddings shape: {embeddings.shape}")


# Check if the number of embeddings matches the number of rows in the data
if embeddings.shape[0] != data.shape[0]:
    logging.error("Mismatch between number of embeddings and number of rows in data.")
    raise ValueError("Mismatch between number of embeddings and number of rows in data.")

# Load the BERT model for generating embeddings (if needed for additional tasks)
model = SentenceTransformer('bert-base-nli-mean-tokens')

# Initialize Hugging Face Inference Client
model_id = "meta-llama/Llama-2-7b-chat-hf"
api_token = os.getenv('HF_API_TOKEN')
if not api_token:
    raise ValueError("Hugging Face API token is not found. Set your HF_API_TOKEN environment variable.")
client = InferenceClient(token=api_token)

# Function to extract sentence embeddings using the BERT model
def extract_sentence_embedding(text):
    logging.debug(f"Extracting embedding for text: {text}")
    return model.encode(text)

# Function to map qualitative terms to dataframe query conditions
def map_qualitative_terms(parameters):
    mappings = {
        'lifelasting battery': 'battery_capacity >= 5000',
        'high performance': 'processor in ["Snapdragon 888", "Apple A14 Bionic"]',
        'good camera': 'rear_camera >= 48',
        'large display': 'display_size >= 6.5',
        'budget friendly': 'discounted_price < 15000'
    }
    conditions = []
    for term, condition in mappings.items():
        for param in parameters.values():
            if re.search(term, param, re.IGNORECASE):
                conditions.append(condition)
    return " and ".join(conditions) if conditions else None

# Function to generate a response using the Hugging Face Inference Client
def generate_response_with_inference(query, recommendation, required_features):
    input_text = (
        f"User query: {query}\n"
        f"Recommended Product: {recommendation}\n"
        f"Required Features: {required_features}\n"
        f"Provide a concise and helpful response based on the recommendation, focusing on how the product meets the required features. "
        f"Response should be around 2 sentences."
    )
    logging.debug(f"Generating response with input text: {input_text}")
    response = client.text_generation(model=model_id, prompt=input_text)
    
    # Check if response is a string or a dictionary
    if isinstance(response, str):
        generated_text = response
    elif 'generated_text' in response:
        generated_text = response['generated_text']
    else:
        logging.error(f"Unexpected response format: {response}")
        return "I'm sorry, I couldn't generate a response."

    # Extract the relevant part of the response (assuming it's the first part of the generated text)
    if generated_text:
        lines = generated_text.split('\n')
        cleaned_response = lines[0].strip() if lines else generated_text.strip()
        return cleaned_response
    return "I'm sorry, I couldn't generate a response."

# Function to get content-based recommendations based on the search query and optional conditions
def get_content_based_recommendations(search_query, qualitative_condition=None, top_n=1):
    search_embedding = extract_sentence_embedding(search_query)
    
    # Normalize embeddings
    normalized_embeddings = embeddings / norm(embeddings, axis=1, keepdims=True)
    normalized_search_embedding = search_embedding / norm(search_embedding)
    
    # Calculate cosine similarity
    scores = np.dot(normalized_embeddings, normalized_search_embedding)
    top_indices = scores.argsort()[-top_n:][::-1]

    # Filter based on qualitative conditions if provided
    if qualitative_condition:
        filtered_data = data.query(qualitative_condition)
        if not filtered_data.empty:
            filtered_indices = [data.index.get_loc(i) for i in filtered_data.index]
            top_indices = [i for i in top_indices if i in filtered_indices]

    if not top_indices:
        return None

    recommendation = data.iloc[top_indices[0]]
    recommendation_data = {
        "brand": recommendation['brand'],
        "model": recommendation['model'],
        "discounted_price": recommendation['discounted_price'],
        "colour": recommendation['colour'],
        "processor": recommendation['processor'],
        "rear_camera": recommendation['rear_camera'],
        "battery_capacity": recommendation['battery_capacity'],
        "battery_type": recommendation['battery_type']
    }
    return recommendation_data

# Define the home route
@app.route('/')
def home():
    return "Server is up and running"

# Defined the webhook route for handling Dialogflow webhook requests
@app.route('/webhook', methods=['POST'])
def webhook():
    req = request.get_json(silent=True, force=True)
    logging.debug(f"Webhook request: {req}")

    # Extract parameters using Llama
    user_query = req['queryResult']['queryText']
    logging.debug(f"User query: {user_query}")

    # Using Llama to understand the user query
    inference_input = (
        f"Extract the following details from the user query if available:\n"
        f"Brand (e.g., Apple, Samsung):\nModel (e.g., iPhone 13, Galaxy S21):\n"
        f"Color (e.g., red, blue):\nBattery_type (e.g., Li-ion, NiMH):\n"
        f"Battery_capacity (e.g., 3000mAh, 4000mAh):\nCamera (e.g., 12MP, 48MP, great camera):\n"
        f"Processor (e.g., Snapdragon 888, A14 Bionic, strong processor):\n"
        f"Battery_quality (e.g., long-lasting, quick charge):\n"
        f"Query: {user_query}"
    )
    llama_response = client.text_generation(model=model_id, prompt=inference_input)
    
    if isinstance(llama_response, dict) and 'generated_text' in llama_response:
        extracted_details = llama_response['generated_text']
    else:
        extracted_details = llama_response

    logging.debug(f"Extracted details using Llama: {extracted_details}")

    # Parse extracted details into parameters
    def parse_detail(key, text):
        match = re.search(fr'{key}:\s*([\w\s-]+)', text, re.IGNORECASE)
        value = match.group(1).strip() if match and match.group(1).strip().lower() != "not available" else ''
        return value

    parameters = {
        "Brand": parse_detail('Brand', extracted_details),
        "Model": parse_detail('Model', extracted_details),
        "color": parse_detail('Color', extracted_details),
        "Battery_type": parse_detail('Battery_type', extracted_details),
        "battery_capacity": parse_detail('Battery_capacity', extracted_details),
        "camera": parse_detail('Camera', extracted_details),
        "processor": parse_detail('Processor', extracted_details),
        "battery_quality": parse_detail('Battery_quality', extracted_details)
    }

    logging.debug(f"Parsed parameters: {parameters}")

    brand = parameters.get('Brand', '').strip()
    color = parameters.get('color', '').strip()
    model_name = parameters.get('Model', '').strip()
    battery_type = parameters.get('Battery_type', '').strip()
    battery_capacity = parameters.get('battery_capacity', '').strip()
    camera = parameters.get('camera', '').strip()
    processor = parameters.get('processor', '').strip()
    battery_quality = parameters.get('battery_quality', '').strip()

    # Check if at least one feature is provided
    if not any([brand, color, model_name, battery_type, battery_capacity, camera, processor, battery_quality]):
        logging.debug("No specific features provided.")
        return jsonify({'fulfillmentText': "Could you please specify? What kind of smartphone would you like?"})

    qualitative_condition = map_qualitative_terms(parameters)
    search_query = " ".join(filter(None, [brand, model_name, color, processor, camera, battery_type, str(battery_capacity), battery_quality]))
    required_features = ", ".join(filter(None, [f"Brand: {brand}", f"Model: {model_name}", f"Color: {color}", f"Processor: {processor}", f"Camera: {camera}", f"Battery Type: {battery_type}", f"Battery Capacity: {battery_capacity}", f"Battery Quality: {battery_quality}"]))
    logging.debug(f"Constructed search query: {search_query}")

    if search_query:
        recommendation = get_content_based_recommendations(search_query, qualitative_condition)
        if recommendation:
            recommendation_text = (
                    f'{recommendation["brand"]} {recommendation["model"]} (Price: {recommendation["discounted_price"]} Euro, '
                    f'Color: {recommendation["colour"]}, Processor: {recommendation["processor"]}, '
                    f'Camera: {recommendation["rear_camera"]} MP, Battery: {recommendation["battery_capacity"]} mAh {recommendation["battery_type"]})'
                )
           
            logging.debug(f"Recommendation: {recommendation_text}")
            response_text = generate_response_with_inference(user_query, recommendation_text, required_features)
            response = {
                'fulfillmentText': response_text,
                'fulfillmentMessages': [
                    {'text': {'text': [response_text]}}
                ]
            }
            logging.debug(f"Response: {response}")
            return jsonify(response)
        else:
            logging.debug("No suitable product found based on the query.")
            return jsonify({'fulfillmentText': "I couldn't find a suitable product based on your query."})
    else:
        logging.debug("No search query formed from parameters.")
        return jsonify({'fulfillmentText': "Could you please specify? What kind of smartphone would you like?"})

# Define a route for sending a message
@app.route('/send-message', methods=['POST'])
def send_message():
    req = request.get_json(silent=True, force=True)
    logging.debug(f"Send message request: {req}")
    text = req.get('message')

    session_id = uuid.uuid4().hex
    session = session_client.session_path(project_id, session_id)

    text_input = dialogflow.TextInput(text=text, language_code='en-US')
    query_input = dialogflow.QueryInput(text=text_input)
    request_dialogflow = dialogflow.DetectIntentRequest(session=session, query_input=query_input)

    try:
        response = session_client.detect_intent(request=request_dialogflow)
        result = response.query_result
        logging.debug(f"Dialogflow response: {result.fulfillment_text}")
        return jsonify({'message': result.fulfillment_text})
    except Exception as e:
        logging.error(f"Dialogflow API error: {e}")
        return jsonify({'error': 'Error connecting to Dialogflow'}), 500

# Run the Flask application
if __name__ == '__main__':
    app.run(port=5000, debug=True)
