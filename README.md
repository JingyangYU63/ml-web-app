# Deploy ML models with Python, FastAPI and Docker
## 1. Files/ folders comments:

ridge_regression.py: The jupyter notebook I used to train my model, downloaded from Google Colab;

app folder: contains an implementation of web application using fastapi;

main.py: creates the web app and implementes GET/POST HTTP methods;

model folder: contains the trained model;

model.py: retrieve the saved model and set up I/O;

trained linear model.pkl: The trained model I saved from ridge_regression.py.

## 2. Relevent links

Github Repository: https://github.com/JingyangYU63/ml-web-app

Dockerhub Repository: https://hub.docker.com/repository/docker/jy732/ml-web-app/general

Heroku App Link: https://ml-web-app.herokuapp.com/

## 3. Model intro

I proposed the method of predicting a future day's receipts number based on the past few (30 in my code) days. For the samll volume of data, I choose the a linear model - Ridge Regression over deep learning/ neural networks (a simple model usually exhibits better performance over deep neural networks when data volume is relatively small). Before building up my model, I choose to read and pre-process the data, including adding a normalization step to control the data value within a common scale (10^6-10^7 is too large). As the optimization method, SGD + momentum helps accelerate gradients vectors in the right directions, thus leading to faster converging. Beside that, I also observed that a good memory locality (where the variable’s memory accesses are
more predictable, i.e.avoid memory allocation in the running of program by pre-allocating memories ahead) could improve performance of SGD. In this project, Bayesian Optimization is adapted for hyperparameter tuning for its efficiency over grid search/ random search (instead of painstakingly trying every hyperparameter set or testing hyperparameter sets at random, the Bayesian optimization method can converge to the optimal hyperparameters. Thus, the best hyperparameters can be obtained without exploring the entire sample space).

## 4. How to run this app?

### i. Run docker in command line

Run commands below in command line to lauch the app (make sure you're under the directory of ml-web-app):
```bash
docker build -t app-name .

docker run -p 80:80 app-name
```
Then open your web browser at your local host port http://0.0.0.0:80/docs. Press the "Try it out" icon under the POST tab and replace the "string" with the month of 2022 you're looking for.
<img width="1382" alt="image" src="https://user-images.githubusercontent.com/73151841/216742437-1125e7a8-e4d9-4f27-b6f0-72d45873bf62.png">
By hitting the execute icon you'll get the estimated number of the scanned receipts for the month you specified at the 	
response body.
<img width="1368" alt="image" src="https://user-images.githubusercontent.com/73151841/216742459-433d66c9-02bc-4bd4-bb86-8ed45897d243.png">

### ii. Send POST request through Postman

First you need to signup with Postman (https://www.postman.com/), then create a new request. Select the POST request and paste the link (https://ml-web-app.herokuapp.com/predict) to the webpage. Then select "Body" tab below and enter the JSON request. Finally, you will get the result by hitting "Send" icon.
<img width="1438" alt="image" src="https://user-images.githubusercontent.com/73151841/216753552-365c2d1b-89f7-4c3b-af5f-3a663e23e933.png">
