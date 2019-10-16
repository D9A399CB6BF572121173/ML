# -*- coding: utf-8 -*-
# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

#getting data
df = pd.read_csv('Path to file/Train+Test_unprocessed.csv',encoding='latin-1')
df.isnull().sum()

#handling missing data and null values
df['Year of Record'].fillna(int(df['Year of Record'].mean()), inplace=True)

df['Gender'].unique()
df['Gender'].replace(['0'], ['unknown'], inplace=True)
df['Gender'].fillna('unknown', inplace=True)

df['Age'].fillna(int(df['Age'].mean()), inplace=True)

df['Profession'].fillna('ProfessionInfoMissing', inplace=True)

df['University Degree'].unique()
df['University Degree'].replace(['0'], ['No'], inplace=True)
df['University Degree'].fillna('DegreeInfoMissing', inplace=True)

#saving dependable variable
y = df.iloc[0:111993, 9].values

del df['Wears Glasses']
del df['Hair Color']
del df['Instance']
del df['Income in EUR']

#one-hot encoding on data
one_hot = pd.get_dummies(df['Gender'])
df = df.drop('Gender',axis = 1)
df = df.join(one_hot)

one_hot = pd.get_dummies(df['University Degree'])
df = df.drop('University Degree',axis = 1)
df = df.join(one_hot)

one_hot = pd.get_dummies(df['Country'])
df = df.drop('Country',axis = 1)
df = df.join(one_hot)

one_hot = pd.get_dummies(df['Profession'])
df = df.drop('Profession',axis = 1)
df = df.join(one_hot)

#separating train and test set
X = df.iloc[0:111827, :].values
X_test=df.iloc[111993:185224, :].values

#Applying Random Forest
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(X, y)

y_pred = regressor.predict(X_test)

