# use curl to test prediction service, grab random rows from test data at ../data/future_unseen_examples.csv
# and send to prediction service
# usage: ./test_predict.sh <port>
# example: ./test_predict.sh 8080

port=${1:-5000}

printf '{
  "bedrooms": 4,
  "bathrooms": 1.0,
  "sqft_living": 1680,
  "sqft_lot": 5043,
  "floors": 1.5,
  "waterfront": 0,
  "view": 0,
  "condition": 4,
  "grade": 6,
  "sqft_above": 1680,
  "sqft_basement": 0,
  "yr_built": 1911,
  "yr_renovated": 0,
  "zipcode": 98118,
  "lat": 47.5354,
  "long": -122.273,
  "sqft_living15": 1560,
  "sqft_lot15": 5765
}' | jq > tmp.json

# test prediction service
curl -d @tmp.json -H "Content-Type: application/json" http://localhost:$port/predict

printf '{
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
  "sqft_lot15": 11875
}' | jq > tmp.json

# test prediction service
curl -d @tmp.json -H "Content-Type: application/json" http://localhost:$port/predict

# clean up
rm tmp.json