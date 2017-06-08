import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.optimizers import SGD, Adam
from keras.utils import np_utils

train_data = pd.read_csv('train.csv/train.csv')
test_data = pd.read_csv('test.csv/test.csv')
train_data = train_data.fillna(0)
test_data = test_data.fillna(0)

data_length = train_data.shape[0]
test_data_length = test_data.shape[0]

train_price = train_data['price_doc']
time = train_data['timestamp']
full_sq = train_data['full_sq'].reshape(data_length,1)
life_sq = train_data['life_sq'].reshape(data_length,1)
floor = train_data['floor'].reshape(data_length,1)
max_floor = train_data['max_floor'].reshape(data_length,1)
build_year = train_data['build_year'].reshape(data_length,1)
kitch_sq = train_data['kitch_sq'].reshape(data_length,1)

test_time = test_data['timestamp']
test_full_sq = test_data['full_sq'].reshape(test_data_length,1)
test_life_sq = test_data['life_sq'].reshape(test_data_length,1)
test_floor = test_data['floor'].reshape(test_data_length,1)
test_max_floor = test_data['max_floor'].reshape(test_data_length,1)
test_build_year = test_data['build_year'].reshape(test_data_length,1)
test_kitch_sq = test_data['kitch_sq'].reshape(test_data_length,1)

X_train = np.concatenate((full_sq,life_sq,floor,max_floor,build_year,kitch_sq),axis=1)
Y_train = train_price
X_test = np.concatenate((test_full_sq,test_life_sq,test_floor,test_max_floor,test_build_year,test_kitch_sq),axis=1)

X_train = np.array(X_train)
X_test = np.array(X_test)

print (X_test.shape)
input_dim = X_train.shape[1]





