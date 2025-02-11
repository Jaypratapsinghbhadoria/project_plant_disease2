import os
import requests
from groq import Groq

client = Groq(
    api_key=os.getenv('GROQ_API_KEY'),
)

def generate_text_from_image(image_path, prediction, confidence):
    # Replace 'your_groq_api_endpoint' and 'your_api_key' with actual values
    api_endpoint = "https://api.groq.com/openai/v1/chat/completions"
    
    with open(image_path, 'rb') as image_file:
        files = {'image': image_file}
        data = {
            'prediction': prediction,
            'confidence': confidence
        }
        headers = {'Authorization': f'Bearer {client.api_key}'}
        response = requests.post(api_endpoint, files=files, data=data, headers=headers)
        
    if response.status_code == 200:
        return response.json().get('generated_text', 'No text generated')
    else:
        print(f"Error: {response.status_code}")
        print(f"Response Content: {response.content}")
        return 'Error in generating text'

def generate_text_from_prediction(prediction, confidence):
    messages = [
        {
            "role": "user",
            "content": f"Explain the prediction '{prediction}' with a confidence of {confidence:.2f}%. Tells about the disease and the plant if it is not healthy. And tell how to cure it.",
        }
    ]
    chat_completion = client.chat.completions.create(
        messages=messages,
        model="llama-3.3-70b-versatile",
    )
    return chat_completion.choices[0].message.content
