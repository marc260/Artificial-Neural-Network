# Artificial Neural Network
# ANN using k-fold cross validation
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

# Data Preprocessing

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')

# Getting independent variables that have an impact on the depended variable (in this case Exited)
x = dataset.iloc[:, 3:13].values # Include Indexes from 3 to 12 (independent variables)
y = dataset.iloc[:, 13].values # (dependent variable)

# Encoding categorical data
# Encodes strings in the data into numbers and etc (in this case, country and gender into 1s and 0s)
# Encoding the Independent Variable
label_encoder_x_country = LabelEncoder() # label for country column
x[:, 1] = label_encoder_x_country.fit_transform(x[:, 1]) # country is the second column (index 1)
label_encoder_x_gender = LabelEncoder() # label for gender column
x[:, 2] = label_encoder_x_gender.fit_transform(x[:, 2]) # gender is the third column (index 2)

# Create dummy variables
# in this case we only need to create dummy variables for country since country has 3 values and gender 2
one_hot_encoder = OneHotEncoder(categorical_features= [1])
x = one_hot_encoder.fit_transform(x).toarray()
x = x[:, 1:] # remove one dummy variable from country, now country and gender have 2 encoded values each

# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 0)

# Feature Scaling
# Necessary since we will have many calculations and we do not want a independent variable dominating another
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Improving the ANN using k-fold cross validation

# Evaluating the ANN
# Building classifier with the same architecture as ann.py
def build_classifier():
    # Initializing the ANN, no layers yet
    classifier = Sequential()

    # Adding input layer and the first hidden layer
    # 1)Number of nodes in the input layer = 11 (number of independent variables input_dim = 11)
    # 2)Number of nodes in the output layer = 1 (dependent variable has 1 or 0 outcome)
    # Avg of 1) and 2) = 11 + 1 / (2) = 6, units=6
    # initialize the weights randomly but close to zero init= 'uniform'
    # Activation function used on the hidden layer is going to be rectifier activation function = 'relu'
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))

    # Adding the second hidden layer, second layer already knows to expect 11 input nodes so input_dim not necessary
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

    # Adding output layer
    # Dependent variable is a categorical number with a binary outcome so units= 1
    # Activation function used on the output layer to get a probability is going to be sigmoid
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

    # Compiling the ANN = applying Stochastic Gradient Descent to the ANN
    # Stochastic Gradient Descent algorithm => optimizer='adam'
    # loss function => logarithmic loss function, dependent variable has a binary outcome so, loss = 'binary_crossentropy'
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

# Create global classifier that will be trained in k-fold cross validation (same batch_size and epochs as ann.py)
classifier = KerasClassifier(build_fn= build_classifier, batch_size= 10, epochs= 100)

class run():
    def __init__(self):
        # Contains the 10 accuracy values returned by cross_val_score
        # estimator = object to use to fir the data
        # X= data to fit, y= target variable to try to predict
        # cv= number of train test fold you want to create
        # n_jobs= number of CPUs to use, -1 = all CPUs parallel computations
        accuracies = cross_val_score(estimator=classifier, X=x_train, y=y_train, cv=10, n_jobs=-1)
        print(accuracies)
        print(accuracies.mean()) #mean
        print(accuracies.std()) #variance
# Only way I've found to run parallel computations on Windows
if __name__ == '__main__':
    run()