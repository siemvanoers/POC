# Text Classification System

This project is a text classification system that uses a machine learning model to categorize text into predefined categories. The model is built using Python and the scikit-learn library.  

## Features
- Load or initialize a text classification model.
- Predict the category of a given text.
- Incrementally learn from user feedback to improve the model.
- Save and load the model and training data.

## Requirements
- Python 3.9 or higher
- scikit-learn
- pickle

## Installation 
### Clone the repository:  
- git clone https://github.com/siemv/POC.git
- cd POC
### Create a virtual environment and activate it:  
- python3 -m venv .venv
- source .venv/bin/activate
### Install the required packages:  
- pip install -r requirements.txt

## Usage
Run the main script:  
- python model.py

Follow the prompts to enter the path to the text file you want to classify.  
The system will predict the category of the text and ask for feedback to improve the model.
