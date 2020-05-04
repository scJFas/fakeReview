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
    model.add(layers.Dense(20, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    #model.add(layers.Softmax())

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
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
    word4 = layers.Dense(20, activation="relu")(word3)
    word5 = layers.Dense(2)(word4)

    input2 = tf.keras.Input(shape=(1, 3), dtype="float64")
    user1 = layers.LSTM(1)(input2)

    input3 = tf.keras.Input(shape=(1, 3), dtype="float64")
    shop1 = layers.LSTM(1)(input3)

    cat = layers.concatenate([word5, user1, shop1])
    flat = layers.Flatten()(cat)

    dense1 = layers.Dense(2, activation="relu")(flat)
    dense2 = layers.Dense(1, activation="sigmoid")(dense1)

    model = tf.keras.Model(inputs=[input1,input2,input3], outputs=dense2)

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model

def model_origin():
    tf.keras.backend.set_floatx('float64')

    input1 = tf.keras.Input(shape=(1, 128, 128), dtype="float64")
    word1 = layers.Conv2D(32, (3, 128), (1, 1), data_format="channels_first", activation="relu")(input1)
    word2 = layers.MaxPool2D((2, 1), data_format="channels_first")(word1)
    word3 = layers.Flatten(data_format="channels_first")(word2)
    word4 = layers.Dense(20, activation="relu")(word3)
    word5 = layers.Dense(2)(word4)

    input2 = tf.keras.Input(shape=(1, 3), dtype="float64")
    user1 = layers.Flatten()(input2)

    input3 = tf.keras.Input(shape=(1, 3), dtype="float64")
    shop1 = layers.Flatten()(input3)

    cat = layers.concatenate([word5, user1, shop1])
    flat = layers.Flatten()(cat)

    dense1 = layers.Dense(2, activation="relu")(flat)
    dense2 = layers.Dense(1, activation='sigmoid')(dense1)

    model = tf.keras.Model(inputs=[input1,input2,input3], outputs=dense2)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model

def main():
    model = complete_model()
    print(model.summary())
    return 0
if __name__ == "__main__":
    main()