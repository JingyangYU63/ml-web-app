# Deploy ML models with Python, FastAPI, Docker and Heroku
## 1. Files/ folders comments:

ridge_regression.ipynb: The original jupyter notebook I used to train my model, downloaded from Google Colab;

ridge_regression.py: The transformed .py format of ridge_regression.ipynb;

app folder: contains an implementation of web application using fastapi;

main.py: creates the web app and implementes GET/POST HTTP methods;

model folder: wraps up the trained model;

model.py: retrieves the saved model and sets up I/O;

trained linear model.pkl: The trained model I saved from ridge_regression.py;

neural network.ipynb: My neural network try out which didn't work out well.

## 2. Relevent links

Github Repository: https://github.com/JingyangYU63/ml-web-app

Dockerhub Repository: https://hub.docker.com/repository/docker/jy732/ml-web-app/general

Heroku App Link: https://ml-web-app.herokuapp.com/

## 3. Model intro

I proposed the method of predicting a future day's receipts number based on the past few (30 in my code) days. For the samll volume of data, I choose a linear model - Ridge Regression over deep learning/ neural networks like LSTM (a simple model usually exhibits better performance over deep neural networks when data volume is relatively small). Before building up my model, I choose to read and pre-process the data, including adding a data normalization step to maintain the data value within a common scale (10^6-10^7 is too large). As the optimization method, SGD + momentum helps accelerate gradients vectors in the right directions, thus leading to faster converging (this is crucial when tuning the hyperparameters which requires high amount of computation). Beside that, I also observed that a good memory locality (where the variable’s memory accesses are more predictable, i.e. avoids memory allocation in the running of program by pre-allocating memories ahead) could improve performance of SGD. In this project, Bayesian Optimization is adapted for hyperparameter tuning for its efficiency over grid search/ random search (instead of painstakingly trying every hyperparameter set or testing hyperparameter sets at random, the Bayesian optimization method can converge to the optimal hyperparameters. Thus, the best hyperparameters can be obtained without exploring the entire sample space).

Additionally, I also used hand-written backprop system for tensors to optimize a simple one-hidden-layer neural network. Similarly, I adapted SGD + momentum as my optimizer and Bayesian Optimization for hyperparameter tuning. However, the prediction result come out bad like I expected. It was quite well when predicting the first few days of 2022, before the predicted figures converge to some constant number when day number increased (the differennce between predctions for Dec 30 and Dec 31 is less than 10). Though I didn't use the prediction made by the neural network when building the web application, I still include the jupyter notebook in the git repository incase you're interested.

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
response body (getting the number of the scanned receipts for June 2022 in the below example).
<img width="1381" alt="image" src="https://user-images.githubusercontent.com/73151841/216753914-cc26d085-8944-44df-9104-1450a50867b5.png">

### ii. Send POST request through Postman (Recommended)

First you need to signup with Postman (https://www.postman.com/), then create a new request. Select the POST request and paste the link (https://ml-web-app.herokuapp.com/predict) to the webpage. Then select "Body" tab below and enter the request body in JSON format. Finally, you will get the result by hitting "Send" icon (getting the number of the scanned receipts for June 2022 in the below example).
<img width="1438" alt="image" src="https://user-images.githubusercontent.com/73151841/216753552-365c2d1b-89f7-4c3b-af5f-3a663e23e933.png">
