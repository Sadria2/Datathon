# ----------------------------------

# EXPLANATION

# This dataset deals with our favourite beverage: wine.
# It is about a white variant of the Portuguese "Vinho Verde".
# Vinho verde is a unique product from the Minho (northwest) region of Portugal.
# Medium in alcohol, it is particularly appreciated due to its freshness (specially in the summer).

# The goal is to model wine quality based on physicochemical tests and sensory data.
# There is no data about grape types, brand, selling price, etc.
# As the class is ordered, you could choose to do a regression instead of a classification.

# The prediction will have to be uploaded in one of your public repository in your github

# Deadline is at 5.30pm!

# For every minute of late delivery your final accuracy will be decreased by 0.05
# For example If your final accuracy is 86% but you deliver 20 minutes late then
# your final accuracy will be 0.86 - (0.05 x 20 = 1) = 0.85
#
# No extra point for early delivery.

# Let's start !

# ----------------------------------

# Import Libraries

import pandas as pd
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.naive_bayes import GaussianNB




# Deleting index columns
traini = pd.read_csv("data/training.csv")
vali = pd.read_csv("data/validation.csv")

traini[(np.abs(stats.zscore(traini)) < 3).all(axis=1)]

#### Removing rows which chloride value is higher than 0.12 and other stuff
traini = traini.drop(traini[traini['volatile.acidity'] > 0.8].index)
traini = traini.drop(traini[traini['residual.sugar'] > 30].index)
#traini = traini.drop(traini[traini['chlorides'] > 0.25].index)
traini = traini.drop(traini[traini['free.sulfur.dioxide'] > 120].index)
#traini = traini.drop(traini[traini['citric.acid'] > 1.6].index)

del traini['Unnamed: 0']
del vali['Unnamed: 0']
del traini['density']
del vali['density']
# Make a correlation matrix
corr = traini.corr()


# split
traini, testi = train_test_split(traini, test_size=0.3, random_state=25)

# Selecting features

features_traini = traini.drop('quality', axis=1)
features_testi = testi.drop('quality', axis=1)
features_vali = vali.drop('quality', axis=1)

# Selecting target variable

target_traini = traini['quality']
target_testi = testi['quality']
target_vali = vali['quality']

# Need some tunning
rf_classifier = RandomForestClassifier()
ros = RandomOverSampler(random_state=42)
gnb = GaussianNB()


# Oversampling

X_res, y_res = ros.fit_resample(features_traini,target_traini)
print('Resampled dataset shape %s' % Counter(y_res))

#Modelling

rf_classifier.fit(features_traini, target_traini)
predictions = rf_classifier.predict(features_testi)
acc = accuracy_score(target_testi, predictions)
print('Accuracy: %.3f' % acc)
cnf_matrix = confusion_matrix(target_testi, predictions)
print('Model with GaussianNB')
y_pred = gnb.fit(features_traini, target_traini).predict(features_testi)
acc3 = accuracy_score(target_testi, predictions)
print('Accuracy: %.3f' % acc3)

print('Model with oversampling:')
rf_classifier.fit(X_res, y_res)
predictions2 = rf_classifier.predict(features_testi)
acc2 = accuracy_score(target_testi, predictions2)
print('Accuracy: %.3f' % acc2)
cnf_matrix2 = confusion_matrix(target_testi, predictions2)

# model for the validation data

finalpredictions = rf_classifier.predict(features_vali)
vali['quality'] = finalpredictions
vali.to_csv('finalpredictions.csv', sep=',')
# plots


fig = go.Figure(data=[go.Histogram(x=traini['quality'])])
#fig.show()

fig = go.Figure(data=[go.Histogram(x=testi['quality'])])
#fig.show()

fig = go.Figure(data=[go.Histogram(x=y_res)])
#fig.show()

#for i in features_traini.columns:
#    fig = go.Figure(data=go.Scatter(x=target_traini, y=features_traini[i], mode='markers'))
#    fig.show()





