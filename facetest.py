from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten,Dropout
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
import tensorflow as tf
from keras.optimizers import SGD

import cv2
import numpy as np
import os
from keras import backend as K
K.set_image_dim_ordering('th')
path = '/home/atefeh/Desktop/computervision/Dataset1/Train'

listing = os.listdir(path)
y_train = [i[1:3] for i in listing]
x_train = []
for i in listing:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(path + '/' + i)
    image = cv2.resize(image, (224, 224)).astype(np.float32)
    image= image/ 255
    x_train.append(image)

# reading the images, preprocessing
# creating X_train and Y_train

x_train = np.array(x_train)
x_train = x_train.reshape(x_train.shape[0],3, 224, 224)

# x_train = tf.convert_to_tensor(x_train, dtype = tf.float32)
y_train = np.array([int(i) for i in y_train])
y_train=y_train-1
input_shape = (3,224, 224)


model = Sequential([
    Conv2D(64, (3, 3), input_shape=input_shape, padding='same', activation='relu'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    Conv2D(128, (3, 3), activation='relu', padding='same', ),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(256, (3, 3), activation='relu', padding='same', ),
    Conv2D(256, (3, 3), activation='relu', padding='same', ),
    Conv2D(256, (3, 3), activation='relu', padding='same', ),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(512, (3, 3), activation='relu', padding='same', ),
    Conv2D(512, (3, 3), activation='relu', padding='same', ),
    Conv2D(512, (3, 3), activation='relu', padding='same', ),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(512, (3, 3), activation='relu', padding='same', ),
    Conv2D(512, (3, 3), activation='relu', padding='same', ),
    Conv2D(512, (3, 3), activation='relu', padding='same', ),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Flatten(),
    Dense(4096, activation='relu'),
    Dense(4096, activation='relu'),
    Dense(4, activation='softmax')
])
model.summary()
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3, steps_per_epoch=9)
# compiling the model?
# fitting the model9
# Test set

path_test ='/home/atefeh/Desktop/computervision/Dataset1/Test'

listing = os.listdir(path_test)
y_test = [i[1:3] for i in listing]
y_test = np.array([int(i) for i in y_test])
y_test=y_test-1
x_test = []

for i in listing:
    # load the image, pre-process it, and store it in the data list
    image_ = cv2.imread(path_test + '/' + i)
    image_ = cv2.resize(image_, (224, 224)).astype(np.float32)
    image_=image_/255
    x_test.append(image_)

x_test = np.array(x_test)
x_test = x_test.reshape(x_test.shape[0], 3, 224, 224)
y_test = np.array([int(i) for i in y_test])

scores = model.evaluate(x_test, y_test, verbose=0)

model1 = Sequential([
    Conv2D(filters=96, input_shape=(3,224,224), kernel_size=(11,11), strides=(4,4), padding="valid",activation='relu'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2),padding='valid'),
    Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding="valid",activation='relu'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2),padding='valid'),
    Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding="valid",activation='relu'),
    Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding="valid",activation='relu'),
    Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="valid",activation='relu' ),
    MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"),
    Flatten(),
    Dense(4096, activation='relu'),
    Dropout(0.4),
    Dense(4096, activation='relu'),
    Dropout(0.4),
    Dense(4, activation='relu'),
    Dropout(0.4),
    Dense(4,activation='softmax')

])
model1.load_weights()
model1.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model1.fit(x_train,y_train,epochs=5, steps_per_epoch=3)
scores = model1.evaluate(x_test, y_test, verbose=0)
