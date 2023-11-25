#!/usr/bin/env python3
import pytest
from flask_testing import TestCase

import sys
from pathlib import Path

# hacky way to import app.py
sys.path.append(str(Path(__file__).resolve().parent.parent))
from app import app


class AppTest(TestCase):
    test_data = {
        "bedrooms": 2,
        "bathrooms": 1.0,
        "sqft_living": 920,
        "sqft_lot": 43560,
        "floors": 1.0,
        "waterfront": 0,
        "view": 0,
        "condition": 4,
        "grade": 5,
        "sqft_above": 920,
        "sqft_basement": 0,
        "yr_built": 1923,
        "yr_renovated": 0,
        "zipcode": 98024,
        "lat": 47.5245,
        "long": -121.931,
        "sqft_living15": 1530,
        "sqft_lot15": 11875,
    }

    def create_app(self):
        app.config["TESTING"] = True
        return app

    def test_index(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

    def test_predict(self):
        response = self.client.post("/predict", json=self.test_data)
        self.assertEqual(response.status_code, 200)
        self.assertIn("prediction", response.json)

    def test_predict_lgbm(self):
        response = self.client.post("/predict_lgbm", json=self.test_data)
        self.assertEqual(response.status_code, 200)
        self.assertIn("prediction", response.json)


if __name__ == "__main__":
    pytest.main()
