from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

def get_model():

    model= Sequential()
    model.add(Dense(800, input_dim=784))
    model.add(Dense(10, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="SGD")
    model.summary()

    return model(inputs=x,outputs=y)