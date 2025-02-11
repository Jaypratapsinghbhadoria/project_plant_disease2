import requests

def generate_text_from_image(image_path, prediction, confidence):
    # Replace 'your_groq_api_endpoint' and 'your_api_key' with actual values
    api_endpoint = "https://api.groq.com/openai/v1/chat/completions"
    api_key = "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"
    
    with open(image_path, 'rb') as image_file:
        files = {'image': image_file}
        data = {
            'prediction': prediction,
            'confidence': confidence
        }
        headers = {'Authorization': f'Bearer {api_key}'}
        response = requests.post(api_endpoint, files=files, data=data, headers=headers)
        
    if response.status_code == 200:
        return response.json().get('generated_text', 'No text generated')
    else:
        return 'Error in generating text'
