from numpy import loadtxt
import pandas as pd
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
import json

dataset = loadtxt('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv', delimiter=',')
dataset.shape
dataset

X = dataset[:, :-1]
Y = dataset[:, -1]

json_file = open('./model.json', 'r')
loaded_model_json = json.load(json_file)
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('model.h5')
print('loaded model from disk')

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y)
print(f'{loaded_model.metrics_names[1]}, {score[1]*100}')
