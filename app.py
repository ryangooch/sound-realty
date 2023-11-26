#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline


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

# load models here to avoid loading them on every request
with open(MODEL_PATH, "rb") as fil:
    KNN_MODEL: Pipeline = pickle.load(fil)

with open(LGBM_MODEL_PATH, "rb") as fil:
    LGBM_MODEL: LGBMRegressor = pickle.load(fil)

# load the demographics data here to avoid loading it on every request
ZIPCODE_DEMOGRAPHICS = pd.read_csv(ZIPCODE_DEMOGRAPHICS_PATH)


@app.route("/")
def index():
    return render_template("index.html")


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
    data: Dict = request.get_json(force=True)

    # Load the model
    model: Pipeline = KNN_MODEL

    demographics: pd.DataFrame = ZIPCODE_DEMOGRAPHICS

    # Convert the JSON object into a dataframe
    df = pd.DataFrame(data, index=[0])  # passing only scalar values, needs index

    # Ensure only desired columns are present
    df = df[INITIAL_MODEL_COLUMNS]

    # Merge the demographics data with the input data
    df = df.merge(demographics, how="left", on="zipcode").drop(columns="zipcode")

    # Make the prediction
    prediction: float = model.predict(df)[
        0
    ]  # model.predict() returns a numpy array, so we select the
    # first element to get the scalar value we want

    # Return the prediction as JSON
    return jsonify({"prediction": prediction})


@app.route("/predict_lgbm", methods=["POST"])
def predict_lgbm():
    data: Dict = request.get_json(force=True)

    model: LGBMRegressor = LGBM_MODEL

    demographics: pd.DataFrame = ZIPCODE_DEMOGRAPHICS

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
    prediction: float = np.exp(model.predict(df)[0])

    return jsonify({"prediction": prediction})


if __name__ == "__main__":
    app.run(debug=True)
