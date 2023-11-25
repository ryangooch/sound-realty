#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify


app = Flask(__name__)
MODEL_PATH = Path("model/model.pkl")
LGBM_MODEL_PATH = Path("model/lgbm_model.pkl")
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


def load_lightgbm_model():
    with open(LGBM_MODEL_PATH, "rb") as fil:
        model = pickle.load(fil)
    return model


@app.route("/")
def index():
    website_str = """
    <h1>Home Sales Price Predictor</h1>
    <p>This app serves a model trained on home sales data via a RESTful API.</p>
    <p>Use the <code>/predict</code> endpoint to make a prediction.</p>
    <p>Example:</p>
    <pre>
    curl -X POST -H "Content-Type: application/json" \\
        -d '{"bedrooms": 3, "bathrooms": 2, "sqft_living": 2000, "sqft_lot": 5000, \\
            "floors": 1, "waterfront": 0, "view": 0, "condition": 3, "grade": 7, \\
            "sqft_above": 2000, "sqft_basement": 0, "yr_built": 1990, \\
            "yr_renovated": 0, "zipcode": "98115", "lat": 47.6809, "long": -122.285, \\
            "sqft_living15": 2000, "sqft_lot15": 5000}' \\
        http://localhost:5000/predict
    </pre>

    <p>Or use the <code>/predict_lgbm</code> endpoint to make a prediction using a
    LightGBM model.</p>
    <p>Example:</p>
    <pre>
    curl -X POST -H "Content-Type: application/json" \\
        -d '{"bedrooms": 3, "bathrooms": 2, "sqft_living": 2000, "sqft_lot": 5000, \\
            "floors": 1, "waterfront": 0, "view": 0, "condition": 3, "grade": 7, \\
            "sqft_above": 2000, "sqft_basement": 0, "yr_built": 1990, \\
            "yr_renovated": 0, "zipcode": "98115", "lat": 47.6809, "long": -122.285, \\
            "sqft_living15": 2000, "sqft_lot15": 5000}' \\
        http://localhost:5000/predict_lgbm
    </pre>
    """
    return website_str


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


@app.route("/predict_lgbm", methods=["POST"])
def predict_lgbm():
    data = request.get_json(force=True)

    model = load_lightgbm_model()

    demographics = pd.read_csv(ZIPCODE_DEMOGRAPHICS_PATH)

    df = pd.DataFrame(data, index=[0])

    # we include date features, so let's assume inference takes place today
    today = datetime.today()

    # break date out into year, month, and day columns
    df["year"] = today.year
    df["month_sin"] = np.sin(2 * np.pi * ((today.month - 1) / 12))
    df["month_cos"] = np.cos(2 * np.pi * ((today.month - 1) / 12))
    df["day_sin"] = np.sin(2 * np.pi * ((today.day - 1) / 31))
    df["day_cos"] = np.cos(2 * np.pi * ((today.day - 1) / 31))

    df = df.merge(demographics, how="left", on="zipcode")

    dropcols = ["zipcode"]
    df = df.drop(columns=dropcols, axis=1)

    # model was trained on log-transformed target
    prediction = np.exp(model.predict(df)[0])

    return jsonify({"prediction": prediction})


if __name__ == "__main__":
    app.run(debug=True)
