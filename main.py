import pandas as pd
import pandas_profiling
from pandas import *
import numpy as np
import sklearn
from pandas_profiling import ProfileReport
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error, ConfusionMatrixDisplay
from sklearn import tree
from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import xgboost as xgb
import matplotlib.pyplot as plt

################################################

# Testing different sklearn classifiers based on diabetes data

################################################
df = read_csv('diabetes_data_upload.csv')
print(df.shape)

#df = df.drop_duplicates()
df = pd.get_dummies(df, drop_first=True)


y = df.Diabetes_Positive
X = df.drop(columns="Diabetes_Positive")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


classifiers = [MLPClassifier(),
               tree.DecisionTreeClassifier(),
               NearestCentroid(),
               SGDClassifier(loss="hinge", penalty="l2", max_iter=5),
               KNeighborsClassifier(3),
               SVC(kernel="linear", C=0.025),
               SVC(gamma=2, C=1),
               GaussianProcessClassifier(1.0 * RBF(1.0)),
               RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
               AdaBoostClassifier(),
               GaussianNB(),
               QuadraticDiscriminantAnalysis()]

names = ['Neural Net (MLPClassifier)',
         'Decision Tree Classifier',
         'Nearest Centroid',
         'Stochastic Gradient Descent',
         'Nearest Neighbors',
         'Linear SVM',
         'RBF SVM',
         'Gaussian Process',
         'Random Forest',
         'AdaBoost',
         'Naive Bayes',
         'QDA'
         ]

score = []

for i, j in enumerate(classifiers):
    clf = j
    clf = clf.fit(X_train, y_train)
    print(names[i])
    print('Correct predictions: ', "{:.2%}".format(clf.score(X_test, y_test)))
    score.append(clf.score(X_test, y_test))
    y_pred = clf.predict(X_test)
    print('Prediction matrix')
    print(confusion_matrix(y_test, y_pred), '\n')
    if i == 1:
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred)


for y, x in enumerate(score):
    score[y] = "{:.4}".format(x)

#print(score)
score2 = []
for i in score:
    score2.append(float(i))
for h, i in enumerate(score2):
    score2[h] = i * 100

fig = plt.figure(figsize=(10, 5))
plt.barh(names, score2, color='blue')
plt.title("Dokładność (Accuracy) klasyfikatorów")
plt.show()











#profile = ProfileReport(df)
#profile.to_file('raport_bez_duplikatow')


