import os
import requests
from groq import Groq

client = Groq(
    api_key='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',
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

def generate_text_from_prediction(prediction, confidence, image_path=None):
    if confidence < 80 and image_path:
        messages = [
            {
                "role": "user",
                "content": "Tell about the leaf in the image and if there is a disease, explain how to cure it.",
            }
        ]
        with open(image_path, 'rb') as image_file:
            files = {'image': image_file.read()}
            chat_completion = client.chat.completions.create(
                messages=messages,
                model="llama-3.3-70b-versatile",
                files=files
            )
    else:
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
