import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing, LinearRegression
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
color = sns.color_palette()

train_data = pd.read_csv('train.csv/train.csv')
test_data = pd.read_csv('test.csv/test.csv')

train_price = train_data['price_doc']
time = train_data['timestamp']

"""
###################################
#  observe the data distribution  #
###################################
plt.figure()
sns.distplot(np.log(train_price), bins=50, kde=True)
plt.xlabel('index')
plt.ylabel('price')
plt.show()

##########################################
#  observe the change of price wrt time  #
##########################################
plt.figure()
sns.barplot(time, train_price, alpha=0.8, color=color[2])
plt.ylabel('Median Price', fontsize=12)
plt.xlabel('Year Month', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()

################################################
#  calculate the missing data of each feature  #
################################################

missing_df = train_data.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df = missing_df.ix[missing_df['missing_count']>0]
ind = np.arange(missing_df.shape[0])
width = 0.9
fig, ax = plt.subplots(figsize=(12,18))
rects = ax.barh(ind, missing_df.missing_count.values, color='y')
ax.set_yticks(ind)
ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')
ax.set_xlabel("Count of missing values")
ax.set_title("Number of missing values in each column")
plt.show()
"""

############################################
#  observe the importance of each feature  #
############################################
"""
for f in train_data.columns:
    if train_data[f].dtype=='object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train_data[f].values)) 
        train_data[f] = lbl.transform(list(train_data[f].values))
        
train_y = train_price
train_X = train_data.drop(["id", "timestamp", "price_doc"], axis=1)

xgb_params = {
    'eta': 0.05,
    'max_depth': 8,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}
dtrain = xgb.DMatrix(train_X, train_y, feature_names=train_X.columns.values)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=100)

# plot the important features #
fig, ax = plt.subplots(figsize=(12,18))
xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
plt.show()
"""



