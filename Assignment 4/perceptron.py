#-------------------------------------------------------------------------
# AUTHOR: Priyanshu Shekhar
# FILENAME: perceptron.py
# SPECIFICATION: Neural network training and testing on optdigits dataset
# FOR: CS 4210- Assignment #4
# TIME SPENT: 4-5 hrs.
#-----------------------------------------------------------*/

# Importing some Python libraries
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd

# Define the values for learning rate and shuffle
n = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
r = [True, False]

df = pd.read_csv('optdigits.tra', sep=',', header=None)

X_training = np.array(df.values)[:, :64]
y_training = np.array(df.values)[:, -1]

df = pd.read_csv('optdigits.tes', sep=',', header=None)

X_test = np.array(df.values)[:, :64]
y_test = np.array(df.values)[:, -1]

highest_perceptron_accuracy = 0.0
highest_mlp_accuracy = 0.0
best_perceptron_params = {'learning_rate': None, 'shuffle': None}
best_mlp_params = {'learning_rate': None, 'shuffle': None}

for learning_rate in n:
    for shuffle in r:
        # Perceptron
        perceptron_clf = Perceptron(eta0=learning_rate, shuffle=shuffle, max_iter=1000)
        perceptron_clf.fit(X_training, y_training)

        perceptron_accuracy = np.mean(perceptron_clf.predict(X_test) == y_test)
        if perceptron_accuracy > highest_perceptron_accuracy:
            highest_perceptron_accuracy = perceptron_accuracy
            best_perceptron_params = {'learning_rate': learning_rate, 'shuffle': shuffle}

        print(f"Highest Perceptron accuracy so far: {highest_perceptron_accuracy:.2f}, Parameters: learning rate={learning_rate}, shuffle={shuffle}")

        # MLP
        mlp_clf = MLPClassifier(activation='logistic', learning_rate_init=learning_rate, hidden_layer_sizes=(25,),
                                shuffle=shuffle, max_iter=1000)
        mlp_clf.fit(X_training, y_training)

        mlp_accuracy = np.mean(mlp_clf.predict(X_test) == y_test)
        if mlp_accuracy > highest_mlp_accuracy:
            highest_mlp_accuracy = mlp_accuracy
            best_mlp_params = {'learning_rate': learning_rate, 'shuffle': shuffle}

        print(f"Highest MLP accuracy so far: {highest_mlp_accuracy:.2f}, Parameters: learning rate={learning_rate}, shuffle={shuffle}")

print("\nBest Perceptron Model:")
print(f"Accuracy: {highest_perceptron_accuracy:.2f}")
print(f"Parameters: learning rate={best_perceptron_params['learning_rate']}, shuffle={best_perceptron_params['shuffle']}")

print("\nBest MLP Model:")
print(f"Accuracy: {highest_mlp_accuracy:.2f}")
print(f"Parameters: learning rate={best_mlp_params['learning_rate']}, shuffle={best_mlp_params['shuffle']}")










