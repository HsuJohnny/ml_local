import numpy as np
import pandas as pd

# 48 x 48 gray picture

def nor(f):
	mean=np.mean(f)
	std=np.std(f)
	return np.floor( (f-mean)*255/max(std,1) )

data = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
data = data.as_matrix()
test = test.as_matrix()

x_label = data[:,0].reshape(data.shape[0], 1) # shape (28709, 1)
train = data[:,1].reshape(data.shape[0], 1)  # shape (28107, 1)
test = test[:,1].reshape(test.shape[0], 1)  # shape (7178, 1)

x_train=[]
x_test=[]
nor_x_train=[]
nor_x_test=[]
# change the data from string to int
for i in range(train.shape[0]):
    x_train.append(np.fromstring(train[i][0], dtype=int, sep=' '))
for i in range(test.shape[0]):
    x_test.append(np.fromstring(test[i][0], dtype=int, sep=' '))

x_train = np.array(x_train)
x_test = np.array(x_test)

x_train = x_train.reshape(train.shape[0],48*48)
x_test = x_test.reshape(test.shape[0],48*48)

print ("normalizing...")
for i in range(train.shape[0]):
	nor_x_train.append(nor(x_train[i]))
print ("finish.")

x_train = x_train.reshape(train.shape[0],48,48,1)
x_test = x_test.reshape(test.shape[0],48,48,1)
