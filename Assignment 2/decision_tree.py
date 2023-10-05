#-------------------------------------------------------------------------
# AUTHOR: Priyanshu Shekhar
# FILENAME: decision_tree.py
# SPECIFICATION: reading three csv files and testing a ID3 model
# FOR: CS 4210- Assignment #2
# TIME SPENT: 5 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import csv
import os,copy


dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']
count = 0
data1,data2,data3 = [],[],[]
for ds in dataSets:

    dbTraining = []
    
    X = []
    Y = []

    #reading the training data in a csv file
    with open(ds, 'r') as csvfile:
         reader = csv.reader(csvfile)
         for i, row in enumerate(reader):
             if i > 0: #skipping the header
                count+=1
                dbTraining.append (row)
                if count <= 8:
                    data1.append(row)
                elif count <= 20:
                    data2.append(row)
                else:
                    data3.append(row)
                     
#transform the original categorical training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3 so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
def transform_feature_to_num(dataset):
    training_class_set = []
    for i in range(len(dataset)):
        for j in range(len(dataset[i])-1):
            if j == 0:
                match dataset[i][j]:
                    case "Young":
                        dataset[i][j] = 1
                    case "Prepresbyopic":
                        dataset[i][j] = 2
                    case "Presbyopic":
                        dataset[i][j] = 3
            elif j == 1:
                match dataset[i][j]:
                    case "Myope":
                        dataset[i][j] = 1
                    case "Hypermetrope":
                        dataset[i][j] = 2
            elif j == 2:
                match dataset[i][j]:
                    case "Yes":
                        dataset[i][j] = 1
                    case "No":
                        dataset[i][j] = 2
            elif j == 3:
                match dataset[i][j]:
                    case "Reduced":
                        dataset[i][j] = 1
                    case "Normal":
                        dataset[i][j] = 2

    for i in range(len(dataset)):
        training_class_set.append(dataset[i][4])
        del dataset[i][4]

    for i in range(len(training_class_set)):
        if training_class_set[i] == "Yes":
            training_class_set[i] = 1
        elif training_class_set[i] == "No":
            training_class_set[i] = 2    
    return dataset,training_class_set

#get the feature data and class data 
data1X,data1Y = transform_feature_to_num(data1)
data2X,data2Y = transform_feature_to_num(data2)
data3X,data3Y = transform_feature_to_num(data3)


#loop your training and test tasks 10 times here
def testing(X,Y):
    overall_accuracy = 0
    for i in range (10):

       #fitting the decision tree to the data setting max_depth=3 & train model based on given training set parameter
       clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=3)
       clf = clf.fit(X, Y)

       #read the test data and add this data to dbTest
       dbTest = ["contact_lens_test.csv"]
        
       for data in dbTest:
        
        #reading test data
        test_data = []
        with open(data, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                if i > 0: #skipping the header
                    test_data.append(row)

        #transform text data
        test_dataX,test_dataY = transform_feature_to_num(test_data)

        #append the predicted class to an array for later comparison between true and predicted array
        predicted_dataY = []
        for instance in (test_dataX):
            class_predicted = clf.predict([instance])[0]
            predicted_dataY.append(class_predicted)

        #compare accuracy true vs predicted
        num_of_correct_prediction = 0
        for i in range(len(test_dataY)):
            if predicted_dataY[i] == test_dataY[i]:
                num_of_correct_prediction+=1

        run_accuracy = num_of_correct_prediction/len(test_dataY)
        overall_accuracy += run_accuracy
        
    average_accuracy = overall_accuracy / 10
    return average_accuracy        
    
#print the average accuracy of this model during the 10 runs (training and test set).
avg_accuracy_set1 = testing(data1X,data1Y)
avg_accuracy_set2 = testing(data2X,data2Y)
avg_accuracy_set3 = testing(data3X,data3Y)

#output result
print("Final accuracy for training on contact_lens_training_1.csv:",avg_accuracy_set1)
print("Final accuracy for training on contact_lens_training_2.csv:",avg_accuracy_set2)
print("Final accuracy for training on contact_lens_training_3.csv:",avg_accuracy_set3)