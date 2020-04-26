import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import joblib

def model_CNN(filters = 32):
    tf.keras.backend.set_floatx('float64')
    model = models.Sequential()
    model.add(tf.keras.Input(shape=(1,128,128), dtype="float64", name="inputLayer"))
    model.add(layers.Conv2D(filters, (3,128), (1,1), data_format="channels_first", activation="relu"))
    model.add(layers.MaxPool2D((2,1), data_format="channels_first"))
    model.add(layers.Flatten(data_format="channels_first"))
    model.add(layers.Dense(100, activation="relu"))
    model.add(layers.Dense(2))
    #model.add(layers.Softmax())

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model

def model_CNN2():
    tf.keras.backend.set_floatx('float64')
    model = models.Sequential()
    model.add(tf.keras.Input(shape=(1, 128, 128), dtype="float64", name="inputLayer"))
    model.add(layers.Conv2D(32, (3, 3), (1, 1), data_format="channels_first", activation="relu"))
    model.add(layers.MaxPool2D((2, 2), data_format="channels_first"))
    model.add(layers.Conv2D(64, (3, 3), (1, 1), data_format="channels_first", activation="relu"))
    model.add(layers.MaxPool2D((2, 2), data_format="channels_first"))
    model.add(layers.Flatten(data_format="channels_first"))
    model.add(layers.Dense(100, activation="relu"))
    model.add(layers.Dense(2))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model

def complete_model(filters = 32):
    tf.keras.backend.set_floatx('float64')

    input1 = tf.keras.Input(shape=(1,128,128), dtype="float64")
    word1 = layers.Conv2D(filters, (3,128), (1,1), data_format="channels_first", activation="relu")(input1)
    word2 = layers.MaxPool2D((2,1), data_format="channels_first")(word1)
    word3 = layers.Flatten(data_format="channels_first")(word2)
    word4 = layers.Dense(2)(word3)

    input2 = tf.keras.Input(shape=(1, 3), dtype="float64")
    user1 = layers.LSTM(1)(input2)

    input3 = tf.keras.Input(shape=(1, 3), dtype="float64")
    shop1 = layers.LSTM(1)(input3)

    cat = layers.concatenate([word4, user1, shop1])
    flat = layers.Flatten()(cat)

    dense1 = layers.Dense(4, activation="relu")(flat)
    output = layers.Dense(2)(dense1)

    model = tf.keras.Model(inputs=[input1,input2,input3], outputs=output)

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model

def main():
    model = complete_model()
    print(model.summary())
    return 0
if __name__ == "__main__":
    main()