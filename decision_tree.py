#-------------------------------------------------------------------------
# AUTHOR: Priyanshu Shekhar
# FILENAME: decision_tree.py
# SPECIFICATION: reads the contact_lens.csv file and outputs a decision tree
# FOR: CS 4210-Assignment #1
# TIME SPENT: 2-3 days
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import csv

db = []
X = []
Y = []

#reading the data in a csv file
with open('contact_lens.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append(row)
         print(row)

# Encoding categorical features into numbers
# For Age: Young = 1, Prepresbyopic = 2, Presbyopic = 3
# For Spectacle: Myope = 1, Hypermetrope = 2
# For Astigmatism: No = 1, Yes = 2
# For Tear Production Rate: Reduced = 1, Normal = 2
for row in db:
    age_mapping = {'Young': 1, 'Prepresbyopic': 2, 'Presbyopic': 3}
    spectacle_mapping = {'Myope': 1, 'Hypermetrope': 2}
    astigmatism_mapping = {'No': 1, 'Yes': 2}
    tear_mapping = {'Reduced': 1, 'Normal': 2}
    X.append([age_mapping[row[0]], spectacle_mapping[row[1]], astigmatism_mapping[row[2]], tear_mapping[row[3]]])
    # Encoding classes: Yes = 1, No = 2
    Y.append(1 if row[4] == 'Yes' else 2)

#fitting the decision tree to the data
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(X, Y)

#plotting the decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'], class_names=['Yes', 'No'], filled=True, rounded=True)
plt.show()
