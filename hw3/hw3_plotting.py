import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras import backend as K
from sklearn.metrics import confusion_matrix

data = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
data = data.as_matrix()
test = test.as_matrix()

x_label = data[:,0].reshape(data.shape[0], 1) # shape (28709, 1)
train = data[:,1].reshape(data.shape[0], 1)  # shape (28107, 1)
test = test[:,1].reshape(test.shape[0], 1)  # shape (7178, 1)

x_train=[]
x_test=[]
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

val_data = x_train[0:3000,:]
val_label = x_label[0:3000].reshape(3000,)
val_label = list(val_label)

x_label = np_utils.to_categorical(x_label, 7)

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

model2.add(Conv2D(144,(3,3), activation='relu'))
model2.add(Conv2D(144,(3,3), activation='relu'))
model2.add(Conv2D(144,(3,3), activation='relu'))

model2.add(Flatten())
model2.add(Dense(units=1000,activation='relu'))
model2.add(Dense(units=7,activation='softmax'))
#model2.summary()

model2.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
model2.load_weights('cnn_model.h5',by_name=False)

# draw confusion matrix
ans = []
val_predict = model2.predict(val_data)
for i in range(val_predict.shape[0]):
    for j in range(val_predict.shape[1]):
        if(val_predict[i][j]==np.amax(val_predict[i])):
            ans.append(j)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

class_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neural']
# Compute confusion matrix
cnf_matrix = confusion_matrix(val_label, ans)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

plt.show()


