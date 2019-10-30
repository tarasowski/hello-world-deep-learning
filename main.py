# Keras is used for developing an devaluation deep learning models
# it wraps Theano and TensorFlow and allows you to define and train netural network models in just few lines of code

from numpy import loadtxt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import json

# we'll use the onset of diabetes dataset a standard ML dataset from UCI ML repository
# it is a binary classification problem (onset of diabetes as 1 or not as 0)

# load as matrix of numbers
# 8 input variables and 1 output variable the last column
dataset = loadtxt('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv', delimiter=',')
dataset.shape
dataset

# we will be learning a model to map rows of input variables `X` to an output variable `y`
# y = f(X)

# input variables `X`:
# number of times pregnant
# plasma glucose concentraion a 2 hours in an oral glucose tolerance test
# diastolic blood pressure (mm Hg)
# tricepts skin fold thickness (mm)
# 2-hour serum insulin (mu U/ml)
# body mass index (weight in kg/(height in m)^2)
# diabetes pedigree function
# age (years)

# output variable `y`:
# class vairable (0 or 1)

# once the data is loaded we can split the data into input and output
X = dataset[:, :-1]
Y = dataset[:, -1]
print(X)
print(Y)

# Define Keras Model
# models in keras are defined as a sequence of layers
# we create a sequntial model and add layers one at a time until we're happy with our network architecture
# 1. ensure the input layer has the right number of input features, the input_dim argument is set to 8
# 2. how do we know the number of layers and their types? Through trial and error
#   * we'll use a fully-conected network structure with three layers
#   * fully connected layers are defined using the Dense class: we can specify the number of nodes in the layer as the first argument, and specify the activation function using the activation argument. 
#   * ReLU on the first two layers and the Sigmoid function in the output layer

# adding each layer:
# * the model expects rows of data with 8 variables (input_dim=8)
# * the first hidden layers has 12 nodes and uses the relu activation function
# * the second hidden layer has 8 nodes and uses the relu activation function
# * the output layer has one node and uses the sigmoid activation function
# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu')) # defines the input or visible layer and the first hidden layer
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

## Compiling the keras model
# it uses numerical libraries under the covers (the so-called backend) such as Theano or TensorFlow
# training the ntwork means finding the best set of weigths to map inputs to outputs in our dataset
# we specify the loss function to use to evaluate a set of weights, the optimizer is used to search trough different weights for the network and any optional metrics we would like to collect and report during training
# we use cross entropy as the loss argument
# the optimizer is the efficient sochastic gradient descent algorithm "adam"
# because it is a classification problem, we will collect and report the classification accuracy, via metrics

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the Keras Model
# now it's time to execute the model on some data
# we can train or fit our model on our loaded data by calling the fit() function on the model
# training occurs over epoch and each epoch is split into batches
# epoch: one pass through all of the rows in the training dataset -> is comprised of one or more batches
# batch: one or more samples considere by the model within an epoch before weights are updated
# a fixed number of itertions through the dataset called epochs, we specify with argument epochs
# a number of dataset rows that are considered before the model weights are updated within each epoch, called the batch size and set using the batch_size argument

# for this problem we will run for a smaller number of epochs (150) and use batch size of 10. That means that each epoch will involve 15 updates to the model weights. This is trial and error approach!

# fit the keras model on the dataset
# this is where the work happens on your CPU or GPU
model.fit(X, Y, epochs=150, batch_size=10)

# Evaluate Keras Model
# We have trained our neural network on the entire dataset and we can evaluate the performance of the network on the same dataset.

# evaluate the keras model
# we'll see how the model works on the current data set, but we don't know how it will work on a new dataset
# in order to do so, you can separate the data into train and test datasets for training and evaluation
# evaluate returns a pair of values (loss, accuracy)
loss, accuracy = model.evaluate(X, Y)
print(f'Loss: {loss} and Accuracy: {accuracy * 100}')

# we would like the loss to go to zero and accuracy to go to 1.0 (e.g. 100%)
# this is not possible for any but the most trivial machine learning problems.
# we will always have some error in our model
# the goal is to choose a model configuration and training configuration that achieves the lowest loss and highest accuracy possible for a given dataset

# Neural networks are a stochastic algorithm, meaning that the same algorithm on the same data can train a different model with different skill each time the code is run. This is a feature not a bug.

# Make predictions

# make probability predictions with the model
predictions = model.predict_classes(X)

for i in range(15):
    print(f'{X[i].tolist()} => {predictions[i]} (expected {Y[i]})')

# Keras separates the concerns of saving your model architecture and saving your model weights
# Model weights are saved to HDF5 format
# Model structure can be saved as YAML / JSON

# Save your model to JSON
# JSON is a simple file format for describing data hierarchically

# serialize model to JSON
##
model_json = model.to_json()

with open('model.json', 'w') as json_file:
   print('saving model to disk')
   json.dump(model_json, json_file) 

# serialize weigths to HDF5
model.save_weights('model.h5')
print('saved weights to disk')

# save architecture + model + weights into a single file
model.save('model_simple.h5')
