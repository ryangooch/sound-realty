# sound-realty
Deployable ML model with Flask and Docker, based on housing data.

## Dependencies
A system with Docker installed.

## Installation
Clone the repo to your machine of choice, and using your command line emulator, enter the top level
directory and run:

```bash
docker-compose up --build
```

This should build the Docker container and start the REST server at `http://localhost:5000`.

If you prefer to change this port, simply edit the `compose.yaml` file and specify the port of your
choice, and edit the `Dockerfile` to expose that port.

## Usage
There is a homepage that is accessible via web browser at `http://localhost:5000/`.
The homepage includes usage examples for making predictions via the REST endpoints, `/predict` and
`/predict_lgbm`.
Additionally, the test scripts in `./sound-realty/tests/test_predict.sh` and 
`./sound-realty/tests/test_predict.sh` contain more context that can be used to make predictions via
the API.

Input data to the API should be provided in a JSON format, where keys correspond to a specific set
of features, and values correspond to one sample from the housing dataset.
The expected features are:

```python
['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'sqft_above', 'sqft_basement', 'zipcode']
```

The feature `zipcode` is used to add various location-based features on the backend but is not used directly in the model.

### K-Nearest Neighbors
The `/predict` endpoint implements a KNN model which follows a [Scikit-Learn `RobustScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html#sklearn-preprocessing-robustscaler) in a pipeline.

The API can be called from a terminal using `curl` as follows:

```bash
curl -d @./data/test_row.json -H "Content-Type: application/json" http://localhost:5000/predict
```

where `test_row.json` can be found in the `data/` directory.

### LightGBM
The `/predict_lgbm` endpoint uses a [LightGBM](https://lightgbm.readthedocs.io/en/latest/index.html)
model and a set of custom, derived features to predict housing prices.
The LightGBM model improves upon the KNN method above in terms of mean absolute precision error
(MAPE) performance, with respect to the KNN technique.
It can be called as above, by simply changing the endpoint.

## Model Training
Please refer to the Jupyter Notebook in [model_training.ipynb](notebooks/model_training.ipynb) for
more information on how the LightGBM model is trained.
It should be noted that this notebook was hosted and trained in a different environment than the
Docker container for the overall repo.
So if you wish to run that code, you will need to install additional dependencies, or stand up a
local virtual environment.

## Notes on Implementation
Refer to [Notes.md](Notes.md) for more information.
