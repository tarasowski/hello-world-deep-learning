from numpy import loadtxt
from keras.models import load_model

dataset = loadtxt('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv', delimiter=',')

X = dataset[:, :-1]
Y = dataset[:, -1]

model = load_model('./model_simple.h5')
model.summary()

score = model.evaluate(X, Y)
print(f'{model.metrics_names[1]}, {score[1]*100}')
