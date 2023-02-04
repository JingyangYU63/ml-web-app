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

I proposed the method of predicting a future day's receipts number based on the past few (30 in my code) days. For the samll volume of data, I choose the a linear model - Ridge Regression over deep learning/ neural networks (a simple model usually exhibits better performance over deep neural networks when data volume is relatively small). Before building up my model, I choose to read and pre-process the data, including adding a normalization step to control the data value within a common scale (10^6-10^7 is too large). As the optimization method, SGD + momentum helps accelerate gradients vectors in the right directions, thus leading to faster converging. Beside that, I also observed that a good memory locality (where the variable’s memory accesses are
more predictable, i.e.avoid memory allocation in the running of program by pre-allocating memories ahead) could improve performance of SGD. In this project, Bayesian Optimization is adapted for hyperparameter tuning for its efficiency over grid search/ random search (instead of painstakingly trying every hyperparameter set or testing hyperparameter sets at random, the Bayesian optimization method can converge to the optimal hyperparameters. Thus, the best hyperparameters can be obtained without exploring the entire sample space).

### 4. How to run this app?

## i.
