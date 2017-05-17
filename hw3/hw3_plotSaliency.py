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

x_train = x_train.reshape(train.shape[0],48,48,1)
x_test = x_test.reshape(test.shape[0],48,48,1)

num_ima = 297
sal_image=x_train[num_ima,:].reshape(1,48,48,1).astype('float64')
original_image = x_train[num_ima,:].reshape(48,48)

x_label = np_utils.to_categorical(x_label, 7)

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

model2.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
model2.load_weights('cnn_model_best.h5',by_name=False)

# saliency map
def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    x /= (K.sqrt(K.mean(K.square(x))) + 1e-5)
    return x
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    x = x.reshape(48,48)
    return x
thre = 0.4
step = 1
input_image=model2.input
result = model2.predict(sal_image)
pred = result.argmax(axis=-1)

loss = K.mean(model2.output[:,pred])

grads = K.gradients(loss, input_image)[0]
grads = normalize(grads)
iterate = K.function([input_image, K.learning_phase()], [grads])
grads_value = iterate([sal_image,1])
grads_value = np.array(grads_value).reshape(48,48)

outImg = sal_image.reshape(48,48)
a=np.mean(outImg)
for x in range(48):
	for y in range(48):
		if(grads_value[x][y]<=thre):
			outImg[x][y]=a
"""
for i in range(20):
    grads_value = iterate([hmap,1])
    hmap += grads_value
"""
#grads_value = deprocess_image(np.array(grads_value))

plt.figure()
plt.imshow(outImg, cmap='gray')
plt.show()
plt.imshow(original_image, cmap='gray')
plt.show()
