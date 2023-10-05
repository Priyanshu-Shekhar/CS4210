#-------------------------------------------------------------------------
# AUTHOR: Priyanshu Shekhar
# FILENAME: naive_bayes.py
# SPECIFICATION: Classify testing data based on training dataset
# FOR: CS 4210- Assignment #2
# TIME SPENT: 5-6 hours
#-----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv,copy

def transform_feature_to_num(dataset):
    training_class_set = []
    day_class_set = []
    for i in range(len(dataset)):
        for j in range(1,len(dataset[i])-1):
            if j == 1:
                match dataset[i][j]:
                    case "Sunny":
                        dataset[i][j] = 1
                    case "Overcast":
                        dataset[i][j] = 2
                    case "Rain":
                        dataset[i][j] = 3
            elif j == 2:
                match dataset[i][j]:
                    case "Hot":
                        dataset[i][j] = 1
                    case "Mild":
                        dataset[i][j] = 2
                    case "Cool":
                        dataset[i][j] = 3
            elif j == 3:
                match dataset[i][j]:
                    case "High":
                        dataset[i][j] = 1
                    case "Normal":
                        dataset[i][j] = 2
            elif j == 4:
                match dataset[i][j]:
                    case "Weak":
                        dataset[i][j] = 1
                    case "Strong":
                        dataset[i][j] = 2

    for i in range(len(dataset)):
        training_class_set.append(dataset[i][5])
        del dataset[i][5]
        day_class_set.append(dataset[i][0])
        del dataset[i][0]

    for i in range(len(training_class_set)):
        if training_class_set[i] == "Yes":
            training_class_set[i] = 1
        elif training_class_set[i] == "No":
            training_class_set[i] = 2    
    return dataset,training_class_set,day_class_set

# These two functions are used to make a nice looking output
def pad_word(word, length):
    word = str(word)
    if len(word) < length:
        padded_word = word + ' ' * (length - len(word))
        return padded_word
    else:
        return word
    
def print_row(c1,c2,c3,c4,c5,c6,c7):
    print(f'{pad_word(c1, 3)}   {pad_word(c2, 8)}   {pad_word(c3, 11)}   {pad_word(c4, 8)}   {pad_word(c5, 6)}   {pad_word(c6, 10)}   {pad_word(c7, 10)}')

#reading the training data in a csv file
db = []

with open('weather_training.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)

#transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]

#get the training feature and training class
X,Y,day_training = transform_feature_to_num(db)

#fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)

#reading the test data in a csv file
test_db = []
with open('weather_test.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         test_db.append (row)

newdb = copy.deepcopy(test_db)

#printing the header os the solution
data_testX,data_testY,day_test = transform_feature_to_num(test_db)

#print headers
print_row("Day","Outlook","Temperature","Humidity","Wind","PlayTennis","Confidence")

count = 0
for instance in data_testX:
    probability_predicted = clf.predict_proba([instance])[0]
    if probability_predicted[0] >= 0.75:
        print_row(newdb[count][0],newdb[count][1],newdb[count][2],newdb[count][3],newdb[count][4],"Yes","%.2f" % probability_predicted[0])
        count+=1
    elif probability_predicted[1] >= 0.75:
        print_row(newdb[count][0],newdb[count][1],newdb[count][2],newdb[count][3],newdb[count][4],"No","%.2f" % probability_predicted[1])
        count+=1
    else:
        count+=1
        
    