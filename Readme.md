# Stroke Detection Machine Learning API

## Introduction
A simple API that uses binary classification algorithms. This API is used to predict whether a patient is likely to get 
stroke based on the input parameters like gender, age, various diseases, and smoking status.

The used classifiers:
 - GradientBoosting
 - KNN
 - RandomForest
 - SVM

This is a learning project where I learnt the basics of machine learning development and experimented a little with binary classification.

## Usage

### Install 
    1. git clone git@github.com:BandiMeszaros/StrokeDetection.git
    2. pip install -r requeierments.txt
    3. python app.py
### Try it out
If everything work as expected you can reach the running api from the browser on http://localhost:5000/docs

### Data
The original data was collected from here: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset and https://www.kaggle.com/datasets/zzettrkalpakbal/full-filled-brain-stroke-dataset?rvi=1. 
The original data is available in the `data/raw_data` folder. The `data/complete_data.csv` contains a concat dataset.
You can use either the `complete_data.csv` file or any of the .csv files in the raw_data folder.


If you would like to test the prediction capabilities of the API you can use any of the .csv files in the `data/demo_test` directory.

## Endpoints
1. **v1/predict**: Takes a .csv file as an input and returns the prediction for all trained models. The input file should contain one entry of patient data.
2. **v1/train**: Takes a .csv file as an input and retrains the models.
