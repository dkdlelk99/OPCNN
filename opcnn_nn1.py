#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.preprocessing import StandardScaler
#from deepchem.feat import molecule_featurizers as fp
from model_0905 import network
from tqdm.notebook import tqdm


# In[2]:


from pandas import read_csv
import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.regularizers import l2
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from keras.constraints import maxnorm
import pandas as pd

from keras import backend as K

# load dataset
# dataframe1 = read_csv("./data/drugbank/ecfp-drugbank-approved.csv", delimiter = ',')
# dataframe2 = read_csv("./data/drugbank/maccskeys-drugbank-approved.csv", delimiter = ',')

dataframe1 = read_csv("./data/drugbank/ecfp-drugbank-all.csv", delimiter = ',')
dataframe2 = read_csv("./data/drugbank/maccskeys-drugbank-all.csv", delimiter = ',')

#dataframe1 = read_csv("ecfp-drugbank-approved.csv", delimiter = ',')
#dataframe2 = read_csv("maccskeys-drugbank-approved.csv", delimiter = ',')
#joind=dataframe1.join(dataframe2, on=None)
dataframe=pd.concat([dataframe1, dataframe2], axis=1)

dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,:]
#Y = np.loadtxt("drugbank_approved_logP.AlogP.value")
# Y = np.loadtxt("./data/drugbank/drugbank_approved_logP.AlogP.value")
Y = np.loadtxt("./data/drugbank/drugbank_all_logP.AlogP.value")


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[3]:


import Split_mcc_nn1
import Split_test_mcc_nn1

import os
import shutil

#import keras
from importlib import reload
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)


# In[4]:


des_col = dataframe1.shape[1]
body_col = dataframe2.shape[1]

reshape_y_train = y_train.reshape(y_train.shape[0],-1)
reshape_y_test = y_test.reshape(y_test.shape[0],-1)

hid_layer = [50, 50, 50]


# In[5]:


reload(keras.models)

learning_rate = 0.001
model_name = 'OPCNN_21_nn1'

for i in tqdm(range(1, 21)) :
    seed_number = 72210300+i

    OPCNN = network(des_col, body_col, hid_layer, 2).OPCNN_21()
    OPCNN = Split_mcc_nn1.CV_Train(X_train, y_train, X_train, y_train, OPCNN, epochs=100,
                                learning_rate=learning_rate, seed_num=seed_number, model_name=model_name)

    pred = OPCNN.base()

    true_pred = pd.concat([pd.DataFrame(reshape_y_train), pd.DataFrame(pred)], axis=1)
    true_pred.columns = ['True', 'Pred', 'Pred_Sigma']
    true_pred.to_csv('./predict/train/{}/predict_all_{}.csv'.format(model_name, seed_number), index=False)


# In[8]:


reload(keras.models)

model_name = 'OPCNN_21_nn1'

for i in tqdm(range(1, 21)) :
    seed_number = 72210300+i

    OPCNN = network(des_col, body_col, hid_layer, 2).OPCNN_21()
    OPCNN = Split_test_mcc_nn1.CV_Test(X_train, y_train, X_test, y_test, OPCNN,
                                    seed_num=seed_number, model_name=model_name)
    pred = OPCNN.base()

    true_pred = pd.concat([pd.DataFrame(reshape_y_test), pd.DataFrame(pred)], axis=1)
    true_pred.columns = ['True', 'Pred', 'Pred_Sigma']
    true_pred.to_csv('./predict/test/{}/predict_all_{}.csv'.format(model_name, seed_number), index=False)


# In[9]:


shutil.rmtree('D:/checkpoint')


# In[ ]:




