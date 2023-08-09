# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 13:53:18 2023

@author: domingosdeeulariadumba
"""




""" IMPORTING LIBRARIES """


# EDA and Plotting

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
plt.style.use('ggplot')


# Machine Learning Modules

from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE


# To save the ML model

import joblib as jb


# For ignoring warnings

import warnings
warnings.filterwarnings('ignore')



"""" EXPLORATORY DATA ANALYSIS """

  
    '''
    Importing the dataset...
    '''
df_bnk = pd.read_csv("C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/ClientConversionPrediction/BankDataset_full.csv",
                 sep = ';')

    '''
    Checking the main info of the dataset...
    '''
df_bnk.info()

    '''
    Statistical summary...
    '''
df_bnk.describe()

    '''
    Presenting the first and last ten entries...
    '''
df_bnk.head(10)
df_bnk.tail(10)

   '''
   Distribution plot.
   For this step we split the attributes into two different lists, one for 
   numerical ('NumList') and other for categorical values ('ObjList').
   '''
ObjList = []
NumList = []

for i in list(df_bnk.columns):
    if df_bnk[i].dtype == 'O':
        ObjList.append(i)
    else:
        NumList.append(i)
 
for j in ObjList[:-1]:
    plt.figure()
    sb.set(rc = {'figure.figsize':(16,12)}, font_scale = 1)
    sb.displot(x = df_bnk[j])
    plt.xticks(rotation = 15)
    plt.savefig ('C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/ClientConversionPrediction/{0}_Cat.barplot.png'.format(j))
    plt.close()

for j in NumList:
    plt.figure()
    sb.set(rc = {'figure.figsize':(16,12)}, font_scale = 1)
    sb.displot(x = df_bnk[j])
    plt.xticks(rotation = 15)
    plt.savefig ('C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/ClientConversionPrediction/{0}_Num.displot.png'.format(j))
    plt.close()

   '''
   Concentration of different attributes on the outcome.
   '''
for q in list(df_bnk.drop('y', axis = 1).columns):
    if np.unique(df_bnk[q]).size < 10:
        plt.figure()
        sb.set(rc = {'figure.figsize':(16,12)}, font_scale = 1)
        pd.crosstab(df_bnk[q], df_bnk.y).plot(kind = 'bar')
        plt.xticks(rotation = 15)
        plt.savefig ('C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/ClientConversionPrediction/{0}_Num.compbarplot.png'.format(q))
        plt.close()
        
   '''
   Conversion Rate Pie Chart.
   '''
conversion = df_bnk['y'].value_counts().reset_index(name = 'conversion')

conv_colors = []

for k in conversion['index']:
    if k == 'no':
        conv_colors.append('red')
    else:
        conv_colors.append('green')
      
plt.figure()
sb.set(rc = {'figure.figsize':(6,6)}, font_scale = 1.5)
conversion.groupby('index').sum().plot(kind = 'pie', colors = conv_colors,
                                     y= 'conversion', autopct='%1.0f%%',
                                     explode = (0.00, 0.15))
plt.savefig ('C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/ClientConversionPrediction/coversion.piechart.png')
plt.close()


    '''
    Mean values of each feature to the campaign result.
    '''
df_bnk.groupby('y').mean()




""" Prediction Model"""



    '''
    Hot encoding and splitting the dataset.
    '''
y = df_bnk['y']
X = pd.get_dummies(df_bnk.drop('y', axis = 1), drop_first = True)

X_train, X_test, y_train, y_test = tts(X,y, test_size = 0.2, random_state = 7)


    '''
    As the pie chart illustrates, the dataset is severely imbalanced. Making
    predictions in this conditions can easily generate biased results (in 
    every 1000 prospects about 120 may subscribe the campaign). To avoid this 
    scenario we'll use an oversampling method: Synthetic Minority Oversampling
    Technique, SMOTE for short.
    '''
X_trainSM, y_trainSM = SMOTE(random_state = 0).fit_resample(X_train, y_train)
print ('Before oversampling:', y_train.value_counts())
print ('After oversampling with SMOTE:', y_trainSM.value_counts())

    '''
    Next is used Recursive Feature Elimination, RFE, to capture the most
    influential attributes so the ML can make more accurate prediction.
    '''
rfeSM = RFE(LogReg(), n_features_to_select = None)
rfeSM_fit = rfeSM.fit(X_trainSM, y_trainSM) 
print(rfeSM.support_)
print(rfeSM.ranking_)

    '''
    Passing the most influential variables to a new set, and printing the final
    attributes...
    '''
X_trainSM_RFE = X_trainSM[X_trainSM.columns[rfeSM.support_.tolist()]]
X_testSM_RFE = X_test[X_trainSM.columns[rfeSM.support_.tolist()]]

print ('Number of final attributes:', len(X_trainSM.columns[rfeSM.support_]))
print ('Final attributes:', X_trainSM.columns[rfeSM.support_])

    '''
    Implementing the model...
    '''
lgreg = LogReg()
lgreg.fit(X_trainSM_RFE, y_trainSM)
y_pred = lgreg.predict(X_testSM_RFE)

print('Accuracy of the model:', lgreg.fit(X_trainSM_RFE, y_trainSM).
                           score(X_testSM_RFE, y_test))

    '''
    Saving the model...
    '''
jb.dump(lgreg,"C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/ClientConversionPrediction/ClientConversion1.sav")
_______________________________________________________________________end________________________________________________________________________