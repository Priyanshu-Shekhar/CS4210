#-------------------------------------------------------------------------
# AUTHOR: Priyanshu Shekhar
# FILENAME: knn.py
# SPECIFICATION: the code 
# FOR: CS 4210- Assignment #2
# TIME SPENT: 4 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv, copy

#function to convert 2D array value to float
def value_to_float_2D(array2D):
    for i in range(len(array2D)):
        for j in range(len(array2D[i])):
            array2D[i][j] = float(array2D[i][j])
    return array2D

def value_to_float_1D(array):
    for i in range(len(array)):
        array[i] = float(array[i])
    return array

db = []
num_of_incorrect_prediction = 0

#reading the data in a csv file
with open('binary_points.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0:
         db.append (row)



#loop your data to allow each instance to be your test set
for data in db:
    #add the training features to the 2D array X removing the instance that will be used for testing in this iteration. For instance, X = [[1, 3], [2, 1,], ...]]. 
    training_db = copy.deepcopy(db)
    training_db.remove(data)
    X = training_db
    training_class_set = []

    for i in range(len(X)):
        training_class_set.append(X[i][2])
        del X[i][2]
    
    #convert 2D str array to float array
    X = value_to_float_2D(X)
    
    #transform the original training classes to numbers and add to the vector Y removing the instance that will be used for testing in this iteration. For instance, Y = [1, 2, ,...]. 
    #1 = positive, 2 = negative
    for i in range(len(training_class_set)):
        if training_class_set[i] == "+":
            training_class_set[i] = 1
        elif training_class_set[i] == "-":
            training_class_set[i] = 2

    Y = training_class_set

    #convert 1D int array to float array
    Y = value_to_float_1D(Y)    

    testSample = data[0:2]
    testSample = value_to_float_1D(testSample)
    test_class = data[2]

    if test_class == "+":
        test_class = 1.0
    elif test_class == "-":
        test_class = 2.0
    
    #fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    #get prediction with testing data
    class_predicted = clf.predict([testSample])[0]

    #compare the prediction with the true label of the test instance to start calculating the error rate.
    if class_predicted != test_class:
        num_of_incorrect_prediction += 1

#print the error rate
error_rate = num_of_incorrect_prediction / len(db)
print("The error rate for 1NN algorithm:",error_rate)