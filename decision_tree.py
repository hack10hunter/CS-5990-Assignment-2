# -------------------------------------------------------------------------
# AUTHOR: Max (Weisheng) Zhang
# FILENAME: decision_tree.py
# SPECIFICATION: Build a decision tree based on cheat_training_1.csv and cheat_training_2.csv, and output the performance of the 2 models.
# FOR: CS 5990 (Advanced Data Mining) - Assignment #2
# TIME SPENT: 1 hour
# -----------------------------------------------------------*/

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from decimal import *

dataSets = ['cheat_training_1.csv', 'cheat_training_2.csv']

for ds in dataSets:

    X = []
    Y = []

    df = pd.read_csv(ds, sep=',', header=0)   #reading a dataset eliminating the header (Pandas library)
    # print("Reading " + ds + "...")
    data_training = np.array(df.values)[:,1:] #creating a training matrix without the id (NumPy library)
    # print(data_training)
    #transform the original training features to numbers and add them to the 5D array X. For instance, Refund = 1, Single = 1, Divorced = 0, Married = 0,
    #Taxable Income = 125, so X = [[1, 1, 0, 0, 125], [2, 0, 1, 0, 100], ...]]. The feature Marital Status must be one-hot-encoded and Taxable Income must
    #be converted to a float.
    # X =
    for row in data_training:
        Xrow = []
        Xrow.append(1 if row[0] == 'Yes' else 2)
        Xrow.append(1 if row[1] == 'Single' else 0)
        Xrow.append(1 if row[1] == 'Divorced' else 0)
        Xrow.append(1 if row[1] == 'Married' else 0)
        Xrow.append(float(row[2][:-1]))
        X.append(Xrow)
    #transform the original training classes to numbers and add them to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    #--> add your Python code here
    # Y =
        Y.append(1 if row[3] == 'Yes' else 2)
    # print("X:")
    # print(X)
    # print("Y:")
    # print(Y)
    #loop your training and test tasks 10 times here
    accuracies = []
    for i in range (10):
       #fitting the decision tree to the data by using Gini index and no max_depth
       clf = tree.DecisionTreeClassifier(criterion = 'gini', max_depth=None)
       clf = clf.fit(X, Y)

       #plotting the decision tree
       tree.plot_tree(clf, feature_names=['Refund', 'Single', 'Divorced', 'Married', 'Taxable Income'], class_names=['Yes','No'], filled=True, rounded=True)
       plt.show()

       #read the test data and add this data to data_test NumPy
       #--> add your Python code here
       # data_test =
       tf = pd.read_csv('cheat_test.csv', sep=',', header=0)
       # print("Reading cheat_test.dsv...")
       data_test = np.array(tf.values)[:,1:]
       
       positive = Decimal('0')
       total = 0
       positive = 0
       for data in data_test:
           #transform the features of the test instances to numbers following the same strategy done during training, and then use the decision tree to make the class prediction. For instance:
           #class_predicted = clf.predict([[1, 0, 1, 0, 115]])[0], where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
           #--> add your Python code here
            total += 1
            row = []
            row.append(1 if data[0] == 'Yes' else 2)
            row.append(1 if data[1] == 'Single' else 0)
            row.append(1 if data[1] == 'Divorced' else 0)
            row.append(1 if data[1] == 'Married' else 0)
            row.append(float(data[2][:-1]))
            # print(row)
            class_predicted = clf.predict([row])[0]
            # print("Class predicted: " + str(class_predicted))
            # print("True label: " + str(data[3]))            
           #compare the prediction with the true label (located at data[3]) of the test instance to start calculating the model accuracy.
           #--> add your Python code here
            if class_predicted == 1 and data[3] == 'Yes' or class_predicted == 2 and data[3] == 'No':
                positive += 1
                # print("Correctly predicted " + str(positive))
       accuracies.append(Decimal(str(positive/total)))  
       # print(accuracies)    
       #find the average accuracy of this model during the 10 runs (training and test set)
       #--> add your Python code here
    averageAccuracy = sum(accuracies)/Decimal('10') 
    #print the accuracy of this model during the 10 runs (training and test set).
    #your output should be something like that: final accuracy when training on cheat_training_1.csv: 0.2
    #--> add your Python code here
    print("final accuracy when training on " + ds + ": " + str(averageAccuracy))


