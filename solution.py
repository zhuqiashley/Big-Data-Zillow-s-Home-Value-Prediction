"""
Zestimate:

Zestimates are estimated home values based on 7.5 million statistical and machine learning models that analyze hunderes of data points on each property. And, by continually imporoving the median margin of error

Objective: Building a model to imporve the Zestimate residual error.

"""


## Step 1: Import Library
#matplotlib inline
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import gc
from sklearn.linear_model import LinearRegression
import random
import datetime
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.layers.noise import GaussianDropout
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer

## Step 1: Import Data

df_train_16 = pd.read_csv('train_2016_v2.csv')
df_train_17 = pd.read_csv('train_2017.csv')
df_prop_16 = pd.read_csv("properties_2016.csv")
df_prop_17 = pd.read_csv("properties_2017.csv")
samplesub = pd.read_csv("sample_submission.csv")

# Step 2: Data Merging for the prediction
data_16 = pd.merge(df_prop_16,df_train_16)
data_17 = pd.merge(df_prop_17,df_train_17)
df = pd.concat([data_16,data_17],keys=('parcelid','transactiondate'))
num_cols = [col for col in df.columns if (df[col].dtype in ['float64','int64'] and col not in ['parcelid','transactiondate']) or df[col].dtype.name=='category']
temp_df = df[num_cols]
for c in df.columns:
    df[c]=df[c].fillna(-1)
    if df[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(df[c].values))
        df[c] = lbl.transform(list(df[c].values))
# Sagrigation of the transcation date into day,month and year wise
df["transactiondate"] = pd.to_datetime(df["transactiondate"])
df["transactiondate_year"] = df["transactiondate"].dt.year
df["transactiondate_month"] = df["transactiondate"].dt.month
df['transactiondate_quarter'] = df['transactiondate'].dt.quarter
df["transactiondate"] = df["transactiondate"].dt.day

# Data is ready now we need to define the data into training and testing data
df = df.fillna(-1.0)
x_train = df.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode','fireplacecnt', 'fireplaceflag'], axis=1)
y_train = df["logerror"]

y_mean = np.mean(y_train)
print(x_train.shape, y_train.shape)
train_columns = x_train.columns

for c in x_train.dtypes[x_train.dtypes == object].index.values:
    x_train[c] = (x_train[c] == True)
samplesub['parcelid'] = samplesub['ParcelId']
df_test = samplesub.merge(df_prop_16 , on='parcelid', how='left')
df_test["transactiondate"] = pd.to_datetime('2016-11-15')  # placeholder value for preliminary version
df_test["transactiondate_year"] = df_test["transactiondate"].dt.year
df_test["transactiondate_month"] = df_test["transactiondate"].dt.month
df_test['transactiondate_quarter'] = df_test['transactiondate'].dt.quarter
df_test["transactiondate"] = df_test["transactiondate"].dt.day     
x_test = df_test[train_columns]

for c in x_test.dtypes[x_test.dtypes == object].index.values:
    x_test[c] = (x_test[c] == True)
# Testing and training data i ready now we need to apply the machine learning model for this
imputer= Imputer()
imputer.fit(x_train.iloc[:, :])
x_train = imputer.transform(x_train.iloc[:, :])
imputer.fit(x_test.iloc[:, :])
x_test = imputer.transform(x_test.iloc[:, :])

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

len_x=int(x_train.shape[1])
print(len_x)
nn = Sequential()
nn.add(Dense(units = 400 , kernel_initializer = 'normal', input_dim = len_x))
nn.add(PReLU())
nn.add(Dropout(.4))
nn.add(Dense(units = 160 , kernel_initializer = 'normal'))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(.6))
nn.add(Dense(units = 64 , kernel_initializer = 'normal'))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(.6))
nn.add(Dense(units = 64 , kernel_initializer = 'normal'))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(.5))
nn.add(Dense(units = 26, kernel_initializer = 'normal'))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(.6))
nn.add(Dense(1, kernel_initializer='normal'))
nn.compile(loss='mae', optimizer=Adam(lr=4e-3, decay=1e-4))
nn.fit(np.array(x_train), np.array(y_train), batch_size = 1000, epochs = 20, verbose=2)
y_pred_ann = nn.predict(x_test)
nn_pred = y_pred_ann.flatten()

pd.DataFrame(nn_pred).head()
y_pred=[]

for i,predict in enumerate(nn_pred):
    y_pred.append(str(round(predict,4)))
y_pred=np.array(y_pred)

output = pd.DataFrame({'ParcelId': df_prop_16['parcelid'].astype(np.int32),
        '201610': y_pred, '201611': y_pred, '201612': y_pred,
        '201710': y_pred, '201711': y_pred, '201712': y_pred})

# set col 'ParceID' to first col
cols = output.columns.tolist()
cols = cols[-1:] + cols[:-1]
output = output[cols]
# Output
print('Output of the model')
print('Output')


