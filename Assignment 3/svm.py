#-------------------------------------------------------------------------
# AUTHOR: Priyanshu Shekhar
# FILENAME: svm.py
# SPECIFICATION: Predict handwritten digits with SVM
# FOR: CS 4210 - Assignment #3
# TIME SPENT: 6 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn import svm
import numpy as np
import pandas as pd

#defining the hyperparameter values
array_c = [1, 5, 10, 100]
array_degree = [1, 2, 3]
array_kernel = ["linear", "poly", "rbf"]
array_decision_function_shape = ["ovo", "ovr"]

df = pd.read_csv('optdigits.tra', sep=',', header=None) #reading the training data by using Pandas library

X_training = np.array(df.values)[:,:64] #getting the first 64 fields to create the feature training data and convert them to NumPy array
y_training = np.array(df.values)[:,-1] #getting the last field to create the class training data and convert them to NumPy array

df = pd.read_csv('optdigits.tes', sep=',', header=None) #reading the training data by using Pandas library

X_test = np.array(df.values)[:,:64] #getting the first 64 fields to create the feature testing data and convert them to NumPy array
y_test = np.array(df.values)[:,-1] #getting the last field to create the class testing data and convert them to NumPy array
highest_accuracy = 0.0
total_prediction = 0
correct_prediction = 0

#created 4 nested for loops that will iterate through the values of c, degree, kernel, and decision_function_shape
for val in array_c:
    for power in array_degree:
        for equation in array_kernel:
           for shape in array_decision_function_shape:

                #Create an SVM classifier that will test all combinations of c, degree, kernel, and decision_function_shape.
                #For instance svm.SVC(c=1, degree=1, kernel="linear", decision_function_shape = "ovo")
                clf = svm.SVC(C=val,degree=power,kernel=equation,decision_function_shape=shape)

                #Fit SVM to the training data
                clf.fit(X_training, y_training)

                #make the SVM prediction for each test sample and start computing its accuracy
                #hint: to iterate over two collections simultaneously, use zip()
                #Example. for (x_testSample, y_testSample) in zip(X_test, y_test):
                #to make a prediction do: clf.predict([x_testSample])
                for (x_testSample,y_testSample) in zip(X_test,y_test):
                    total_prediction += 1
                    y_prediction = clf.predict([x_testSample])
                    #compare y_prediction 1D array to y_testSample
                    if int(y_prediction[0]) == int(y_testSample):
                        correct_prediction += 1
                

                #check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together
                #with the SVM hyperparameters. Example: "Highest SVM accuracy so far: 0.92, Parameters: a=1, degree=2, kernel= poly, decision_function_shape = 'ovo'
                accuracy = correct_prediction / total_prediction
                if accuracy > highest_accuracy:
                    highest_accuracy = accuracy
                    print('Highest SVM accuracy so far: %.5f' % highest_accuracy, 'Parameters: C =',val,'degree =',power,'kernel =',equation,'decision_function_shape =',shape)



