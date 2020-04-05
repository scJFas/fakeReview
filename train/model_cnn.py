import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.externals import joblib

def model_CNN():
    model = models.Sequential()
    model.add(tf.keras.Input(shape=(128,128,1), dtype="float64", name="inputLayer"))
    model.add(layers.Conv2D(10, (3,128), (1,1), activation="relu"))
    model.add(layers.MaxPool2D((2,1)))
    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation="relu"))
    model.add(layers.Dense(2))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model

def main():
    return 0
if __name__ == "__main__":
    main()