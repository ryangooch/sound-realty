#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
from pathlib import Path
from typing import Dict

import pandas as pd
from flask import Flask, request, jsonify, render_template


app = Flask(__name__)
MODEL_PATH = Path("model/model.pkl")
ZIPCODE_DEMOGRAPHICS_PATH = Path("data/zipcode_demographics.csv")
INITIAL_MODEL_COLUMNS = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "sqft_above",
    "sqft_basement",
    "zipcode",
]


def load_model():
    with open(MODEL_PATH, "rb") as fil:
        model = pickle.load(fil)
    return model


@app.route("/")
def index():
    return "This app serves a model trained on home sales data via a RESTful API."


@app.route("/predict", methods=["POST"])
def predict():
    """
    Make a prediction and return it. This function is an endpoint on a RESTful service which
    receives JSON POST data, makes a prediction, and returns the result as JSON, with some relevant
    metadata. The input columns we expect are:
    - bedrooms
    - bathrooms
    - sqft_living
    - sqft_lot
    - floors
    - waterfront
    - view
    - condition
    - grade
    - sqft_above
    - sqft_basement
    - yr_built
    - yr_renovated
    - zipcode
    - lat
    - long
    - sqft_living15
    - sqft_lot15

    The output columns we return are:
    - price
    """
    data = request.get_json(force=True)

    # Load the model
    model = load_model()

    # Load the demographics data
    demographics = pd.read_csv(ZIPCODE_DEMOGRAPHICS_PATH)

    # Convert the JSON object into a dataframe
    df = pd.DataFrame(data, index=[0])  # passing only scalar values, needs index

    # Ensure only desired columns are present
    df = df[INITIAL_MODEL_COLUMNS]

    # Merge the demographics data with the input data
    df = df.merge(demographics, how="left", on="zipcode").drop(columns="zipcode")

    # Make the prediction
    prediction = model.predict(df)[0]  # model.predict() returns a numpy array, so we select the
    # first element to get the scalar value we want

    # Return the prediction as JSON
    return jsonify({"prediction": prediction})


if __name__ == "__main__":
    app.run(debug=True)
