# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

heart = pd.read_csv('./../data/heart.csv') 
le = LabelEncoder()
heart['target'] = le.fit_transform(heart['target'])
heartX = heart.drop('target', 1).copy().values
heartY = heart['target'].copy().values


trainX, testX, trainY, testY = train_test_split(heartX, heartY, test_size=0.3, random_state=0, stratify=heartY)     
trainY = np.atleast_2d(trainY).T
testY = np.atleast_2d(testY).T  
train_X, validation_X, train_Y, validation_Y = train_test_split(trainX, trainY, test_size=0.2, random_state=1, stratify=trainY)
train = pd.DataFrame(np.hstack((train_X, train_Y)))
test = pd.DataFrame(np.hstack((testX, testY)))
validation = pd.DataFrame(np.hstack((validation_X, validation_Y)))
train.to_csv('./../data/heart_train.csv',index=False,header=False)
test.to_csv('./../data/heart_test.csv',index=False,header=False)
validation.to_csv('./../data/heart_validation.csv',index=False,header=False)