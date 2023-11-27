#-------------------------------------------------------------------------
# AUTHOR: Priyanshu Shekhar
# FILENAME: deep_learning.py
# SPECIFICATION: Neural network training and testing on Fashion MNIST dataset
# FOR: CS 4210 - Assignment #4
# TIME SPENT: 3 hrs.
#-----------------------------------------------------------*/

# Importing the libraries
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def build_model(n_hidden, n_neurons_hidden, n_neurons_output, learning_rate):
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[28, 28]))  # input layer

    # Iterate over the number of hidden layers to create the hidden layers
    for _ in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons_hidden, activation="relu"))  # hidden layer with ReLU activation function

    # Output layer
    model.add(keras.layers.Dense(n_neurons_output, activation="softmax"))  # output layer with one neural for each class and softmax activation function

    # Defining the learning rate
    opt = keras.optimizers.SGD(learning_rate)

    # Compiling the Model specifying the loss function and the optimizer to use.
    model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model

# Using Keras to Load the Dataset.
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Iterate over number of hidden layers, number of neurons in each hidden layer, and the learning rate.
n_hidden_values = [2, 5, 10]
n_neurons_values = [10, 50, 100]
learning_rate_values = [0.01, 0.05, 0.1]

highest_accuracy = 0.0
best_model = None

for n_hidden in n_hidden_values:
    for n_neurons in n_neurons_values:
        for learning_rate in learning_rate_values:
            model = build_model(n_hidden, n_neurons, 10, learning_rate)

            history = model.fit(X_train, y_train, epochs=5, validation_data=(X_valid, y_valid))

            # Calculate the accuracy of this neural network and store its value if it is the highest so far.
            accuracy = np.max(history.history['val_accuracy'])
            if accuracy > highest_accuracy:
                highest_accuracy = accuracy
                best_model = model

            print("Highest accuracy so far: " + str(highest_accuracy))
            print("Parameters: " + "Number of Hidden Layers: " + str(n_hidden) + ", Number of neurons: " +
                  str(n_neurons) + ", Learning rate: " + str(learning_rate))
            print()

# After generating all neural networks, print the summary of the best model found
print(best_model.summary())

# Plotting the learning curves of the best model
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)  # set the vertical range to [0-1]
plt.show()

# Save the architecture plot to an image file
img_file = './model_arch.png'
tf.keras.utils.plot_model(best_model, to_file=img_file, show_shapes=True, show_layer_names=True)



