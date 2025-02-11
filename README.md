# Plant Disease Classifier

This project is a web application that classifies plant diseases from images of plant leaves. It uses a pre-trained TensorFlow model to predict the disease and provides additional information using the Groq API.

## Features

- Upload an image of a plant leaf to detect diseases.
- Displays the predicted class and confidence level.
- Provides additional information about the disease and how to cure it using the Groq API.

## Requirements

- Python 3.7+
- TensorFlow
- Streamlit
- Pillow
- NumPy
- Requests
- Groq API Key

## Setup

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/project_plant_disease.git
    cd project_plant_disease
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

3. Set up the Groq API key:
    ```sh
    export GROQ_API_KEY=your_groq_api_key
    ```

4. Run the Streamlit app:
    ```sh
    streamlit run app.py
    ```

## Usage

1. Open the web application in your browser.
2. Upload an image of a plant leaf.
3. View the predicted class and confidence level.
4. Read the additional information about the disease and how to cure it.

## Files

- `app.py`: Main application file.
- `groq_api.py`: Contains functions to interact with the Groq API.
- `requirements.txt`: List of required Python packages.
- `README.md`: Project documentation.

