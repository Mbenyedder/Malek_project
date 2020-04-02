# Import the libraires
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# to avoifd "your CPU supports instructions that this Tf was not compiled to use :AVX2"
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data import read_test_data
from train import read_train_data
from model import get_model


train = read_train_data()
x_train = train.iloc[:, 1:]  # toutes les lignes et toutes les colonnes sauf la 1ere
y_train = train.iloc[:, :1]

y_train = np_utils.to_categorical(y_train, 10)
x_train = x_train / 255

test = read_test_data()

model = Sequential()
model.add(Dense(800, input_dim=784))
model.add(Dense(10, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="SGD")
model.summary()

model.fit(x_train, y_train)

test = test / 255
result = model.predict(test)
y_test = np.argmax(result, axis=1)

Sample_submissions_CNN = pd.DataFrame({"ImageID": range(1, len(y_test) + 1), "Label": y_test})
print(Sample_submissions_CNN.head())
# larger batches = faster training
# batch size is the number of samples that will pass through the network at the same time
Sample_submissions_CNN.to_csv("sub", index=False)

