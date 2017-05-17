import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.optimizers import SGD, Adam
from keras.utils import np_utils

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

x_train = x_train.reshape(train.shape[0],48,48,1)
x_test = x_test.reshape(test.shape[0],48,48,1)
super_x_train = x_train[10000:,:]
semi_x_train = x_train[0:10000,:]
super_x_label = x_label[10000:]
#x_label = np_utils.to_categorical(x_label, 7)
super_x_label_0 = np_utils.to_categorical(super_x_label, 7)

model2 = Sequential()
model2.add(BatchNormalization(input_shape=(48,48,1)))
model2.add(Conv2D(36,(3,3),activation='relu'))
model2.add(Conv2D(36,(3,3),activation='relu'))
#model2.add(Conv2D(32,(3,3),activation='relu'))
model2.add(MaxPooling2D(pool_size=(2,2)))

model2.add(Conv2D(72,(3,3), activation='relu'))
model2.add(Conv2D(72,(3,3), activation='relu'))
model2.add(Conv2D(72,(3,3),activation='relu'))
model2.add(MaxPooling2D(pool_size=(2,2)))

#model2.add(BatchNormalization(input_shape=(48,48,1)))
model2.add(Conv2D(144,(3,3), activation='relu'))
model2.add(Conv2D(144,(3,3), activation='relu'))
model2.add(Conv2D(144,(3,3), activation='relu'))

model2.add(Flatten())
model2.add(Dense(units=1000,activation='relu'))
model2.add(Dense(units=7,activation='softmax'))
model2.summary()

model2.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
model2.fit(super_x_train,super_x_label_0,batch_size=100,epochs=1)

score = model2.evaluate(super_x_train,super_x_label)
print ('\nTrain Acc:', score[1])
#model2.save('cnn_model.h5')
semi_pred = model2.predict(semi_x_train)
ans=[]
for i in range(semi_pred.shape[0]):
    for j in range(semi_pred.shape[1]):
        if(semi_pred[i][j]==np.amax(semi_pred[i])):
            ans.append((i, j))
ans = np.array(ans).reshape(10000,1)
semi_label = np.concatenate((ans,super_x_label), axis=0)
model2.fit(x_train,semi_label,batch_size=100,epochs=1,initial_epoch=2)

pred = model2.predict(x_test)
testAns=[]
for i in range(pred.shape[0]):
    for j in range(pred.shape[1]):
        if(pred[i][j]==np.amax(pred[i])):
            testAns.append((i, j))
name = ["id", "label"]
testAns = pd.DataFrame.from_records(testAns, columns=name)
testAns.to_csv("semi_predict.csv", index=False)



