# -------------------------------------------------------------------------
# AUTHOR: Max (Weisheng) Zhang
# FILENAME: roc_curve.py
# SPECIFICATION: Read cheat_data.csv, split the training and test data and compute the ROC curve for a decision tree classifier.
# FOR: CS 5990 (Advanced Data Mining) - Assignment #2
# TIME SPENT: 20 minutes
# -----------------------------------------------------------*/

#importing some Python libraries
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

# read the dataset cheat_data.csv and prepare the data_training numpy array
# --> add your Python code here
# data_training = ?
df = pd.read_csv('cheat_data.csv', sep=',', header=0)
data_training = np.array(df.values)[:,:]
print(data_training)
# transform the original training features to numbers and add them to the 5D array X. For instance, Refund = 1, Single = 1, Divorced = 0, Married = 0,
# Taxable Income = 125, so X = [[1, 1, 0, 0, 125], [0, 0, 1, 0, 100], ...]]. The feature Marital Status must be one-hot-encoded and Taxable Income must
# be converted to a float.
# --> add your Python code here
# X = ?
X = []
y = []
for row in data_training:
    Xrow = []
    Xrow.append(1 if row[0] == 'Yes' else 0)
    Xrow.append(1 if row[1] == 'Single' else 0)
    Xrow.append(1 if row[1] == 'Divorced' else 0)
    Xrow.append(1 if row[1] == 'Married' else 0)
    Xrow.append(float(row[2][:-1]))
    X.append(Xrow)
# transform the original training classes to numbers and add them to the vector y. For instance Yes = 1, No = 0, so Y = [1, 1, 0, 0, ...]
# --> add your Python code here
# y = ?
    y.append(1 if row[3] == 'Yes' else 0)
# print(X)
# print(y)
# split into train/test sets using 30% for test
# --> add your Python code here
trainX, testX, trainy, testy = train_test_split(X, y, test_size = 0.3)
# generate a no skill prediction (random classifier - scores should be all zero)
# --> add your Python code here
# ns_probs = ?
ns_probs = [0] * len(testy)
# print(ns_probs)
# fit a decision tree model by using entropy with max depth = 2
clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=2)
clf = clf.fit(trainX, trainy)
# tree.plot_tree(clf, feature_names=['Refund', 'Single', 'Divorced', 'Married', 'Taxable Income'], class_names=['Yes','No'], filled=True, rounded=True)
# plt.show()

# predict probabilities for all test samples (scores)
dt_probs = clf.predict_proba(testX)

# keep probabilities for the positive outcome only
# --> add your Python code here
# dt_probs = ?
# print(clf.classes_)
# print(dt_probs)
dt_probs = dt_probs[:, 1]
# print(dt_probs)
# calculate scores by using both classifeirs (no skilled and decision tree)
ns_auc = roc_auc_score(testy, ns_probs)
dt_auc = roc_auc_score(testy, dt_probs)

# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Decision Tree: ROC AUC=%.3f' % (dt_auc))

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
dt_fpr, dt_tpr, _ = roc_curve(testy, dt_probs)

# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(dt_fpr, dt_tpr, marker='.', label='Decision Tree')

# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')

# show the legend
pyplot.legend()

# show the plot
pyplot.show()