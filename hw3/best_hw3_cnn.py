import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.optimizers import SGD, Adam
from keras.utils import np_utils

data = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
test_label = pd.read_csv("predict_best.csv")
data = data.as_matrix()
test = test.as_matrix()
test_label = test_label.as_matrix()

x_label = data[:,0].reshape(data.shape[0], 1) # shape (28709, 1)
train = data[:,1].reshape(data.shape[0], 1)  # shape (28107, 1)
test = test[:,1].reshape(test.shape[0], 1)  # shape (7178, 1)
t_label = test_label[:,1].reshape(test_label.shape[0], 1)

x_train=[]
x_test=[]

# change the data from string to int
for i in range(train.shape[0]):
    x_train.append(np.fromstring(train[i][0], dtype=int, sep=' '))
for i in range(test.shape[0]):
    x_test.append(np.fromstring(test[i][0], dtype=int, sep=' '))

x_train = np.array(x_train)
x_test = np.array(x_test)
x_train = x_train.reshape(train.shape[0],48,48,1)
x_test = x_test.reshape(test.shape[0],48,48,1)

allData = np.concatenate((x_train,x_test), axis=0)
allLabel = np.concatenate((x_label,t_label), axis=0)

x_label = np_utils.to_categorical(x_label, 7)
allLabel = np_utils.to_categorical(allLabel, 7)

model2 = Sequential()
model2.add(BatchNormalization(input_shape=(48,48,1)))
model2.add(Conv2D(36,(3,3),activation='relu'))
model2.add(Conv2D(36,(3,3),activation='relu'))
model2.add(MaxPooling2D(pool_size=(2,2)))

model2.add(Conv2D(72,(3,3), activation='relu'))
model2.add(Conv2D(72,(3,3), activation='relu'))
model2.add(Conv2D(72,(3,3),activation='relu'))
model2.add(MaxPooling2D(pool_size=(2,2)))

model2.add(Conv2D(144,(3,3), activation='relu'))
model2.add(Conv2D(144,(3,3), activation='relu'))
model2.add(Conv2D(144,(3,3), activation='relu'))

model2.add(BatchNormalization(input_shape=(48,48,1)))
model2.add(Flatten())
model2.add(Dense(units=1000,activation='relu'))
model2.add(Dense(units=7,activation='softmax'))
model2.summary()

model2.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
model2.fit(allData,allLabel,batch_size=100,epochs=15)
model2.save_weights('cnn_model_best.h5')

score = model2.evaluate(allData,allLabel)
print ('\nTrain Acc:', score[1])
ans=[]
result = model2.predict(x_test)
for i in range(result.shape[0]):
    for j in range(result.shape[1]):
        if(result[i][j]==np.amax(result[i])):
            ans.append((i, j))

name = ["id", "label"]
ans = pd.DataFrame.from_records(ans, columns=name)
ans.to_csv("predict_best.csv", index=False)



