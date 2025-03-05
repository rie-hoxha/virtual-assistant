# P1 - Enhancing E-Commerce with AI-Powered Shopping Support - Rie Hoxha

## Description
In today's highly competitive e-commerce landscape, providing personalized shopping experiences is crucial. This project aims to enhance the e-commerce experience through the implementation of an AI-powered shopping support system. The focus is on developing a product recommender chatbot with natural language processing (NLP) capabilities. The system will enable users to interact with a chatbot, leveraging NLP to understand and respond to user queries related to product preferences and recommendations. The recommender engine will utilize algorithms such as content-based filtering to generate personalized product suggestions. The web app is simply designed as a chat where users can type in their requests and the Shopping Assistant will respond back. The goal of this project is solely to provide product recommendations. The data is taken from publicly available product datasets: flipkart smartphones in Kaggle. Technologies used for the web app are Frontend -> React and Backend -> Python (Flask, Django) and Chatbot frameworks (DialogFlow). I am also using a large language model such as Llama for NLP output generation and parameter extraction. For the backend, I am using Poetry to create a virtual environment, which is for the purpose of easy running on another machine.

## Functionality Overview
1. **User Interaction**: The chatbot interacts with users through a web interface, allowing them to ask questions about smartphones.
2. **Natural Language Processing**: It uses NLP to understand user queries and extract relevant details such as brand, model, color, battery type, and more.
3. **Product Recommendations**: The chatbot generates product recommendations based on user queries using a content-based filtering approach. It utilizes precomputed embeddings and a dataset of smartphones to find the best matches.
4. **Dialogflow Integration**: The chatbot integrates with Dialogflow to handle user intents and provide responses based on the extracted parameters.
5. **Logging and Error Handling**: The application includes logging for debugging and error handling to ensure smooth operation.
## How to Run the Project
1. Start by running the backend:
   - Use `poetry install` to install all dependencies and then run the main file.
   - Use the command: `ngrok http 5000` to expose an endpoint for the backend on the web.
   - Use the ngrok link to pass it to the Dialogflow fulfillment API to connect them together.
2. Run the frontend to interact with the chatbot.

## Confidential Information Instructions
The project requires certain confidential information, such as API tokens and credentials for services like Hugging Face and Dialogflow. Users should set these as environment variables in a `.env` file. Ensure that the following variables are included:
- `HF_API_TOKEN`: Your Hugging Face API token.
- `GOOGLE_APPLICATION_CREDENTIALS`: Path to your Dialogflow service account JSON file.
