# Deploy ML models with Python, FastAPI and Docker
### 1. Files/ folders comments:

ridge_regression.py: The jupyter notebook I used to train my model, downloaded from Google Colab;

app folder: contains an implementation of web application using fastapi;

main.py: creates the web app and implementes GET/POST HTTP methods;

model folder: contains the trained model;

model.py: retrieve the saved model and set up I/O;

trained linear model.pkl: The trained model I saved from ridge_regression.py.

### 2. Relevent links

Dockerhub Repository: https://hub.docker.com/repository/docker/jy732/ml-web-app/general

### 3. Model intro

For the samll volume of data, I choose the a linear model - Ridge Regression over deep learning/ neural networks (a simple model usually exhibits better performance over deep neural networks when data volume is relatively small). Before building up my model, I choose to read and pre-process the data, including adding a normalization step to reduce the data value to a moderate range ($10^6-10^7$ is too large).
