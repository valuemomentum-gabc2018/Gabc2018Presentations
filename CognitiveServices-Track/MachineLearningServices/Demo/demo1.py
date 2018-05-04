# Use the Azure Machine Learning data preparation package
from azureml.dataprep import package

# Use the Azure Machine Learning data collector to log various metrics
from azureml.logging import get_azureml_logger
from azureml.dataprep.package import run
from sklearn.preprocessing import Imputer
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import sys
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
import os
import pickle

# initialize the logger
run_logger = get_azureml_logger()

# create the outputs folder
os.makedirs('E:/Azure Boot Camp/Project/Demo Project/Demo1/Demo1/outputs', exist_ok=True)

# This call will load the referenced package and return a DataFrame.
# If run in a PySpark environment, this call returns a
# Spark DataFrame. If not, it will return a Pandas DataFrame.

# Loading the Titanic Dataset and Obtaining the shape of data
titanic_train = run('demo1.dprep', dataflow_idx=0, spark=False)
#titanic_test = run('demo1_test.dprep', dataflow_idx=0, spark=False)

print ('Titanic dataset shape: {}'.format(titanic_train.shape))

#explore missing data
titanic_train.apply(lambda x : sum(x.isnull()))

#pre-process Embarked
titanic_train['Embarked'].replace('r^\s+$',np.NaN,regex=True,inplace=True)
titanic_train['Embarked'].replace(np.NaN,'S',regex=True,inplace=True)


#pre-process Age
imp = Imputer(missing_values="NaN",strategy='mean',axis=0)
imp.fit(titanic_train[['Age']])
titanic_train['Age'] = imp.fit_transform(titanic_train[['Age']]).ravel()


#create family size feature

def size_to_type(x):

    if(x == 1): 

        return 'Single'

    elif(x >= 2 and x <= 4): 

        return 'Small'

    else: 

        return 'Large'

    

titanic_train['FamilySize'] = titanic_train.SibSp + titanic_train.Parch + 1

titanic_train['FamilyType'] = titanic_train['FamilySize'].map(size_to_type)

#process names of passengers

title_Dictionary = {

                        "Capt":       "Officer", "Col":        "Officer",

                        "Major":      "Officer", "Jonkheer":   "Royalty",

                        "Don":        "Royalty", "Sir" :       "Royalty",

                        "Dr":         "Officer", "Rev":        "Officer",

                        "the Countess":"Royalty","Dona":       "Royalty",

                        "Mme":        "Mrs", "Mlle":       "Miss",

                        "Ms":         "Mrs", "Mr" :        "Mr",

                        "Mrs" :       "Mrs", "Miss" :      "Miss",

                        "Master" :    "Master", "Lady" :      "Royalty"

}



def extract_title(name):

    return name.split(',')[1].split('.')[0].strip()

titanic_train['Title'] = titanic_train['Name'].map(extract_title)

titanic_train['Title'] = titanic_train['Title'].map(title_Dictionary)

def extract_number(ticket):
    if ticket.split(' '):
       return ticket.split(' ')[-1].strip()
    else :
     return ticket


titanic_train['Ticket_num'] = titanic_train['Ticket'].map(extract_number)

titanic_train1 = pd.get_dummies(titanic_train, columns=['Pclass', 'FamilyType', 'Embarked', 'Sex','Title'])

titanic_train1 = titanic_train1[titanic_train1.Ticket_num != 'LINE']
titanic_train1.Ticket_num = titanic_train1.Ticket_num.astype(np.int64)

Y = titanic_train1['Survived']

titanic_train1.drop(['PassengerId','Name','Cabin','Survived','Ticket'], axis=1, inplace=True)

X = titanic_train1

# split data 70%-30% into training set and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=0)

print(X_train.info())

# change regularization rate and you will likely get a different accuracy.
reg = 0.01
# load regularization rate from argument if present
if len(sys.argv) > 1:
    reg = float(sys.argv[1])

print("Regularization rate is {}".format(reg))

# log the regularization rate
run_logger.log("Regularization Rate", reg)

clf1 = LogisticRegression(C = 1/reg, random_state=0)
clf1.fit(X_train,Y_train)
print (clf1)

# evaluate the test set
accuracy = clf1.score(X_test, Y_test)
run_logger.log("Accuracy", accuracy)

y_pred1 = clf1.predict(X_test) # Prediction

print(confusion_matrix(Y_test,y_pred1))

y_scores = clf1.predict_proba(X_test)
precision, recall, thresholds = precision_recall_curve(Y_test, y_scores[:,1])
run_logger.log("Precision", precision)
run_logger.log("Recall", recall)
run_logger.log("Thresholds", thresholds)

# serialize the model on disk in the special 'outputs' folder
print ("Export the model to model.pkl")
f = open('E:/Azure Boot Camp/Project/Demo Project/Demo1/Demo1/outputs/model.pkl', 'wb')
pickle.dump(clf1, f)
f.close()
