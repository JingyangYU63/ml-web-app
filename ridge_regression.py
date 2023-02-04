#!/usr/bin/env python3
import os
import sys

# BEGIN THREAD SETTINGS this sets the number of threads used by numpy in the program
# (should be set to 1 to avoid implicit parallelism)
implicit_num_threads = 1
os.environ["OMP_NUM_THREADS"] = str(implicit_num_threads)
os.environ["MKL_NUM_THREADS"] = str(implicit_num_threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(implicit_num_threads)
# END THREAD SETTINGS

import numpy
import torch
from numpy import random
import scipy
import matplotlib
import pickle
matplotlib.use('agg')
from matplotlib import pyplot as plt
import threading
import time
import pandas
from sklearn.model_selection import train_test_split
from collections import deque

from tqdm import tqdm
from google.colab import files, drive

drive.mount('/content/gdrive')

path = "/content/gdrive/MyDrive/data_daily.csv"
# reading the CSV file
csvFile = pandas.read_csv(path)
 
# displaying the contents of the CSV file
print(csvFile)

# setting the days ahead range for predicting the approximate number of the scanned receipts for a future day
day_range = 30 # using the data from day 0 to day 29 to predict day 30

# constructing features and target variables
Receipt_Count = csvFile["Receipt_Count"].array
Xs = [[Receipt_Count[j] for j in range(i, i + day_range)] for i in range(335)]
Ys = Receipt_Count[day_range:]
assert len(Xs) == len(Ys)
# perform train-validation (0.8 vs 0.2) split
Xs_tr, Xs_va, Ys_tr, Ys_va = train_test_split(Xs, 
                                              Ys, 
                                              test_size = 0.2, 
                                              random_state = 123,)

# normalization of input data
mean = numpy.mean(Receipt_Count)
std = numpy.std(Receipt_Count)
Xs_tr = numpy.array(Xs_tr, dtype=float).reshape(day_range, -1)
Xs_va = numpy.array(Xs_va, dtype=float).reshape(day_range, -1)
Ys_tr = numpy.array(Ys_tr, dtype=float).reshape(1, -1)
Ys_va = numpy.array(Ys_va, dtype=float).reshape(1, -1)
Xs_tr = (Xs_tr - mean) / std
Ys_tr = (Ys_tr - mean) / std
Xs_tr = (Xs_va - mean) / std
Ys_tr = (Ys_va - mean) / std
receipts_dataset = (Xs_tr, Xs_va, Ys_tr, Ys_va)
print(Xs_tr.shape, Xs_va.shape, Ys_tr.shape, Ys_va.shape)

# weight matrix initialization
W0 = numpy.zeros((len(Ys_tr), len(Xs_tr)))
print(W0.shape)

# SGD + Momentum (threaded)
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# beta            momentum hyperparameter
# B               minibatch size
# num_epochs      number of epochs (passes through the training set) to run
# monitor_period  how frequently, in terms of batches (not epochs) to output the parameter vector
# num_threads     how many threads to use
#
# returns         the final model arrived at at the end of training
def sgd_mss_with_momentum_threaded(Xs, Ys, gamma, W0, alpha, beta, B, num_epochs, num_threads):
    (d, n) = Xs.shape
    (c, d) = W0.shape
    # perform global setup/initialization/allocation
    V = numpy.zeros(W0.shape)
    W = numpy.copy(W0)
    gradient = numpy.zeros(W0.shape)
    Bt = int(B / num_threads)

    # construct the barrier object
    iter_barrier = threading.Barrier(num_threads + 1)

    # a function for each thread to run
    def thread_main(ithread):
        # perform any per-thread allocations
        # avoid memory allocation in the running of program by pre-allocating memories ahead
        XdotX = numpy.zeros((d, d))
        WdotXdotX = numpy.zeros(W0.shape)
        YdotX = numpy.zeros(W0.shape)
        gammaW = numpy.zeros(W0.shape)
        multinomial_logreg_grad_i = numpy.zeros(W0.shape)

        slices_X = []
        slices_Y = []
        for ibatch in range(int(n/B)):
            ii = range(ibatch*B + ithread*Bt, ibatch*B + (ithread+1)*Bt)
            slices_X.append(numpy.ascontiguousarray(Xs[:,ii]))
            slices_Y.append(numpy.ascontiguousarray(Ys[:,ii]))
        # gradint calculation (uses only pre-allocated memories to improve performance of SGD)
        for it in range(num_epochs):
            for ibatch in range(int(n/B)):
                # work done by thread in each iteration;
                # this section of code primarily uses numpy operations with the "out=" argument specified
                numpy.dot(slices_X[ibatch], numpy.transpose(slices_X[ibatch]), out=XdotX)
                numpy.dot(W, XdotX, out=WdotXdotX)
                numpy.dot(slices_Y[ibatch], numpy.transpose(slices_X[ibatch]), out=YdotX)
                numpy.multiply(gamma, W, out=gammaW)
                numpy.subtract(WdotXdotX, YdotX, out=multinomial_logreg_grad_i)
                numpy.add(multinomial_logreg_grad_i, gammaW, out=multinomial_logreg_grad_i)
                
                iter_barrier.wait() # wait for all threads to finish computation before moving up to next step
                numpy.add(gradient, multinomial_logreg_grad_i, out=gradient)
                
                iter_barrier.wait()

    worker_threads = [threading.Thread(target=thread_main, args=(it,)) for it in range(num_threads)]

    for t in worker_threads:
        print("running thread ", t)
        t.start()

    print("Running minibatch sequential-scan SGD with momentum (%d threads)" % num_threads)
    # gradient & momentum update:
    # v <- beta * v - alpha * gradient
    # w <- w + v
    for it in tqdm(range(num_epochs)):
        for ibatch in range(int(n/B)):
            numpy.multiply(gradient, 0, out=gradient)
            iter_barrier.wait()
            # work done on a single thread at each iteration;
            # this section of code primarily uses numpy operations with the "out=" argument specified
            numpy.divide(gradient, B, out=gradient)
            numpy.multiply(beta, V, out=V)
            numpy.multiply(alpha, gradient, out=gradient)
            numpy.subtract(V, gradient, out=V)
            numpy.add(W, V, out=W)
            iter_barrier.wait()

    for t in worker_threads:
        t.join()

    print("current loss: " + str((W @ Xs - Ys) @ (W @ Xs - Ys).T + gamma * W @ W.T)) # report current loss
    # return the learned model
    return W

# customized hyperparameter tryout
sgd_mss_with_momentum_threaded(Xs=Xs_tr, Ys=Ys_tr, gamma=0.0001, W0=W0, alpha=0.001, beta=0.9, B=8, num_epochs=200, num_threads=8)

# gradient descent to do the inner optimization step of Bayesian optimization
#
# objective     the objective function to minimize, as a function that takes a torch tensor and returns an expression
# x0            initial value to assign to variable (torch tensor)
# alpha         learning rate/step size
# num_iters     number of iterations of gradient descent
#
# returns     (obj_min, x_min), where
#       obj_min     the value of the objective after running iterations of gradient descent
#       x_min       the value of x after running iterations of gradient descent
def gradient_descent(objective, x0, alpha, num_iters):
    x = x0.detach().clone()  # create a fresh copy of x0
    x.requires_grad = True   # make it a target for differentiation
    opt = torch.optim.SGD([x], alpha)
    for it in range(num_iters):
        opt.zero_grad()
        f = objective(x)
        f.backward()
        opt.step()
    x.requires_grad = False  # make x no longer require gradients
    return (float(f.item()), x)

# compute the Gaussian RBF kernel matrix for a vector of data points (in PyTorch)
#
# Xs        points at which to compute the kernel (size: d x m)
# Zs        other points at which to compute the kernel (size: d x n)
# gamma     gamma parameter for the RBF kernel
#
# returns   an (m x n) matrix Sigma where Sigma[i,j] = K(Xs[:,i], Zs[:,j])
def rbf_kernel_matrix(Xs, Zs, gamma):
    m = Xs.shape[1] if len(Xs.shape) > 1 else 1
    n = Zs.shape[1] if len(Zs.shape) > 1 else 1
    sigma = [[torch.exp(-gamma * torch.linalg.norm(Xs[:, i] - Zs[:, j])**2) for j in range(n)] for i in range(m)]
    sigma = torch.tensor(sigma)
    return sigma

# compute the distribution predicted by a Gaussian process that uses an RBF kernel (in PyTorch)
#
# Xs            points at which to compute the kernel (size: d x n) where d is the number of parameters
# Ys            observed value at those points (size: n)
# gamma         gamma parameter for the RBF kernel
# sigma2_noise  the variance sigma^2 of the additive gaussian noise used in the model
#
# returns   a function that takes a value Xtest (size: d) and returns a tuple (mean, variance)
def gp_prediction(Xs, Ys, gamma, sigma2_noise):
    # first, do any work that can be shared among predictions
    sigma = rbf_kernel_matrix(Xs, Xs, gamma)
    n = Xs.shape[1]
    # next, define a nested function to return
    def prediction_mean_and_variance(Xtest):
        # construct mean and variance
        k = [torch.exp(-gamma * torch.linalg.norm(Xs[:, i] - Xtest)) for i in range(n)]
        k = torch.tensor(k)
        
        mean = k @ torch.linalg.inv(sigma + sigma2_noise * torch.eye(n)) @ Ys
        variance = torch.exp(-gamma * torch.linalg.norm(Xtest - Xtest)) + sigma2_noise -\
        k @ torch.linalg.inv(sigma + sigma2_noise * torch.eye(n)) @ k.T
        return (mean.reshape(()), variance.reshape(()))
    #finally, return the nested function
    return prediction_mean_and_variance

# run Bayesian optimization to minimize an objective
#
# objective     objective function; takes a torch tensor, returns a python float scalar
# d             dimension to optimize over
# gamma         gamma to use for RBF hyper-hyperparameter
# sigma2_noise  additive Gaussian noise parameter for Gaussian Process
# acquisition   acquisition function to use (e.g. ei_acquisition)
# random_x      function that returns a random sample of the parameter we're optimizing over (a torch tensor, e.g. for use in warmup)
# gd_nruns      number of random initializations we should use for gradient descent for the inner optimization step
# gd_alpha      learning rate for gradient descent
# gd_niters     number of iterations for gradient descent
# n_warmup      number of initial warmup evaluations of the objective to use
# num_iters     number of outer iterations of Bayes optimization to run (including warmup)
#
# returns       tuple of (y_best, x_best, Ys, Xs), where
#   y_best          objective value of best point found
#   x_best          best point found
#   Ys              vector of objective values for all points searched (size: num_iters)
#   Xs              matrix of all points searched (size: d x num_iters)
def bayes_opt(objective, d, gamma, sigma2_noise, acquisition, random_x, gd_nruns, gd_alpha, gd_niters, n_warmup, num_iters):
    y_best = float("inf")
    x_best = torch.zeros(size=(d,))
    Xs = []
    Ys = []
    # warm-up to prepare prior information for Bayesian Optimization
    for _ in range(n_warmup):
        x_i = torch.tensor([0.0001, 0.001, 0.9])
        y_i = objective(x_i)
        Xs.append(x_i)
        Ys.append(y_i)
        if y_i <= y_best:
            y_best = y_i
            x_best = x_i
    for _ in range(n_warmup, num_iters):
        Xs_vec = torch.stack(tensors=Xs, dim=1)
        Ys_vec = torch.tensor(Ys)
        prediction_fn =  gp_prediction(Xs_vec, Ys_vec, gamma, sigma2_noise)
        y = float("inf")
        x = torch.zeros(size=(d,))
        for _ in range(gd_nruns):
            x_0 = random_x(size=(d,))
            _, x_i = gradient_descent(objective=lambda x: acquisition(y_best, prediction_fn(x)[0], torch.sqrt(prediction_fn(x)[1])),\
                x0=x_0, alpha=gd_alpha, num_iters=gd_niters)
            y_i = objective(x_i)
            if y_i <= y:
                y = y_i
                x = x_i
        Xs.append(x)
        Ys.append(y)
        if y <= y_best:
            y_best = y
            x_best = x
    Xs_vec = torch.stack(tensors=Xs, dim=1)
    Ys_vec = torch.tensor(Ys)
    return y_best, x_best, Ys_vec, Xs_vec

# return a function that computes the lower confidence bound (LCB) acquisition function
#
# kappa     parameter for LCB
#
# returns   function that computes the LCB acquisition function
def lcb_acquisition(kappa):
    def A_lcb(Ybest, mean, stdev):
        return mean - kappa * stdev
    return A_lcb

# produce a function that runs SGD+Momentum on the receipts dataset, initializing the weights to zero
#
# mnist_dataset         the MNIST dataset, as returned by load_MNIST_dataset_with_validation_split
# num_epochs            number of epochs to run for
# B                     the batch size
#
# returns               a function that takes parameters
#   params                  a numpy vector of shape (3,) with entries that determine the hyperparameters, where
#       gamma = params[0]
#       alpha = params[1]
#       beta = params[2]
#                       and returns (the validation error of the final trained model after all the epochs) minus 0.9.
#                       if training diverged (i.e. any of the weights are non-finite) then return 0.1, which corresponds to an error of 1.
def receipts_dataset_sgd_mss_with_momentum(receipts_dataset, B, num_epochs, num_threads):
    def objective(params):
        Xs_tr, Xs_va, Ys_tr, Ys_va = receipts_dataset
        d = Xs_tr.shape[0]
        c = Ys_tr.shape[0]
        if torch.is_tensor(Xs_tr):
            Xs_tr = Xs_tr.numpy()
        if torch.is_tensor(Ys_tr):
            Ys_tr = Ys_tr.numpy()
        if torch.is_tensor(Xs_va):
            Xs_va = Xs_va.numpy()
        if torch.is_tensor(Ys_va):
            Ys_va = Ys_va.numpy()
        gamma, alpha, beta, W_0 = float(params[0].item()), float(params[1].item()), float(params[2].item()), numpy.zeros(shape=(c,d))
        W = sgd_mss_with_momentum_threaded(Xs=Xs_tr, Ys=Ys_tr, gamma=gamma, W0=W0, alpha=alpha, beta=beta,\
                                           B=B, num_epochs=num_epochs, num_threads=num_threads)
        Ys_pr = W @ Xs_va
        error = (numpy.sum((Ys_va - Ys_pr)**2) / numpy.sum((numpy.mean(Ys_va) - Ys_va)**2)) # use 1 - R^2 as error to select hyperparameters
        return float(error)
    return objective

# perform Bayesian Optimization to find optimal hyperparameters
obj = receipts_dataset_sgd_mss_with_momentum(receipts_dataset, B=8, num_epochs=40, num_threads=8)
(y_best, x_best, Ys_vec, Xs_vec) = bayes_opt(objective=obj, d=3, gamma=10, sigma2_noise=0.001, acquisition=lcb_acquisition(kappa=2.0),\
                                     random_x=torch.randn, gd_nruns=20, gd_alpha=0.01, gd_niters=20, n_warmup=3, num_iters=20)
print(y_best) # best R^2 score
print(x_best) # best hyperparameter set
print(Ys_vec) # R^2 score history
print(Xs_vec) # hyperparameter set history

# the weight matrix generated under the optimal hyperparameter set
W = sgd_mss_with_momentum_threaded(Xs=Xs_tr, Ys=Ys_tr, gamma=0.1612, W0=W0, alpha=0.0375, beta=0.3463, B=8, num_epochs=40, num_threads=8)

# use the model to predict the approximate number of the scanned receipts for each day of 2022
q = deque()
warm_up = [Receipt_Count[j] for j in range(364, 364 - day_range, -1)]
for count in warm_up:
    q.append(count)
predictions = []
for _ in range(365):
    X = numpy.array(q, dtype=float).reshape(30,)
    predictions.append(((W @ (X - mean) / std) * std + mean).item())
    q.popleft()
    q.append(predictions[-1])
# calculate the predicted number of the scanned receipts for each month of 2022
df = pandas.DataFrame()
df.index = pandas.date_range(start='1/1/2022', end='12/31/2022')
df["count"] = predictions
monthSum = df.groupby(df.index.month).sum()

df # predicted data

# save the trained linear model
with open('trained linear model.pkl','wb') as f:
    pickle.dump(monthSum, f)
