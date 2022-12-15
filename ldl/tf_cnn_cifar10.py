import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
import numpy as np
import logging

tf.get_logger().setLevel(logging.ERROR)

EPOCHS = 128
BATCH_SIZE = 32

# Epoch time: CPU: 12s
# Epoch time: GPU: 8s

cifar_dataset = keras.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = cifar_dataset.load_data()

# standardize the data
mean = np.mean(train_images)
stddev = np.std(train_images)

train_images = (train_images - mean) / stddev
test_images = (test_images - mean) / stddev

print('mean: ', mean)
print('stddev: ', stddev)

# one-hot encode the labels
train_labels = to_categorical(train_labels, num_classes=10)
test_labels = to_categorical(test_labels, num_classes=10)


model = Sequential()
model.add(Conv2D(64, (5, 5), strides=(2, 2), activation='relu', padding='same', input_shape=(32, 32, 3),
          kernel_initializer='he_normal', bias_initializer='zeros'))
model.add(Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding='same',
          kernel_initializer='he_normal', bias_initializer='zeros'))
model.add(Flatten())
model.add(Dense(10, activation='softmax', kernel_initializer='glorot_uniform', bias_initializer='zeros'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

history = model.fit(train_images, train_labels,
                    validation_data=(test_images, test_labels),
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    verbose=2,
                    shuffle=True)
