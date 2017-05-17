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
import time

data = pd.read_csv("train.csv")
data = data.as_matrix()

x_label = data[:,0].reshape(data.shape[0], 1) # shape (28709, 1)
train = data[:,1].reshape(data.shape[0], 1)  # shape (28107, 1)

x_train=[]
# change the data from string to int
for i in range(train.shape[0]):
    x_train.append(np.fromstring(train[i][0], dtype=int, sep=' '))
x_train = np.array(x_train)

x_train = x_train.reshape(train.shape[0],48,48,1)

num_ima = 297
sal_image=x_train[num_ima,:].reshape(1,48,48,1).astype('float64')
original_image = x_train[num_ima,:].reshape(48,48)

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

step = 1
input_img=model2.input
layer_dict = dict([(layer.name, layer) for layer in model2.layers[1:]])

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)
img_width = 48
img_height = 48
layer_name = 'conv2d_1'
kept_filters = []
for filter_index in range(0, 36):
    # we scan through the first 36 filters,
    print('Processing filter %d' % filter_index)
    start_time = time.time()

    # we build a loss function that maximizes the activation
    # of the nth filter of the layer considered
    layer_output = layer_dict[layer_name].output
    if K.image_data_format() == 'channels_first':
        loss = K.mean(layer_output[:, filter_index, :, :])
    else:
        loss = K.mean(layer_output[:, :, :, filter_index])

    # we compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, input_img)[0]

    # normalization trick: we normalize the gradient
    grads = normalize(grads)

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_img, K.learning_phase()], [loss, grads])

    # step size for gradient ascent
    step = 1.

    # we start from a gray image with the chosen image
    input_img_data = sal_image
    """
    if K.image_data_format() == 'channels_first':
        input_img_data = np.random.random((1, 1, img_width, img_height))
    else:
        input_img_data = np.random.random((1, img_width, img_height, 1))
    input_img_data = (input_img_data - 0.5) * 20 + 128
    """
    # we run gradient ascent for 20 steps
    for i in range(20):
        loss_value, grads_value = iterate([input_img_data, 1])
        input_img_data += grads_value * step

        print('Current loss value:', loss_value)
        if loss_value <= 0.:
            # some filters get stuck to 0, we can skip them
            break

    # decode the resulting input image
    if loss_value > 0:
        img = deprocess_image(input_img_data[0])
        kept_filters.append((img, loss_value))
    end_time = time.time()
    print('Filter %d processed in %ds' % (filter_index, end_time - start_time))

# we will stich the 36 filters on a 6 x 6 grid.
n = 5

# the filters that have the highest loss are assumed to be better-looking.
# we will only keep the top 36 filters.
kept_filters.sort(key=lambda x: x[1], reverse=True)
kept_filters = kept_filters[:n * n]

margin = 5
width = n * img_width + (n - 1) * margin
height = n * img_height + (n - 1) * margin
stitched_filters = np.zeros((width, height))

# fill the picture with our saved filters
for i in range(n):
    for j in range(n):
        img, loss = kept_filters[i * n + j]
        stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
                         (img_height + margin) * j: (img_height + margin) * j + img_height] = img

plt.figure()
plt.imshow(stitched_filters, cmap='gray')
plt.show()
