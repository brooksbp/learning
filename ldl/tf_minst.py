import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import numpy as np
import logging

tf.get_logger().setLevel(logging.ERROR)

tf.random.set_seed(7)

# need special conda env to install specific versions of python, tf, cudatoolkit, cudnn for gpu...
devices = tf.config.experimental.list_physical_devices()

EPOCHS = 20

BATCH_SIZE = 1

# CPU: i7-6700K (oneDNN AVX AVX2)
#
# batch size /  epoch     val   val  
# num batches   duration  loss  acc
#
# 1 / 60000     23s       .009  .943
# 8 / 7500       3s       .015  .921
# 64 / 938       1s       .055  .683
# 512 / 118      0s       .089  .341
# 4096 /15       0s       .160  .161

# GPU: GTX1070
#
# batch size /  epoch     val   val  
# num batches   duration  loss  acc
#
# 1 / 60000    106s       .009  .942
# 8 / 7500      15s       .015  .922
# 64 / 938       2s       .061  .647
# 512 / 118      0.4s     .089  .296
# 4096 /15       0.14s    .169  .148

minst = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = minst.load_data()

# standardize the data
mean = np.mean(train_images)
stddev = np.std(train_images)
train_images = (train_images - mean) / stddev
test_images = (test_images - mean) / stddev

# one-hot encode labels
train_labels = to_categorical(train_labels, num_classes=10)
test_labels = to_categorical(test_labels, num_classes=10)

initializer = keras.initializers.RandomUniform(minval=-0.1, maxval=0.1)

# Flatten: 28x28 -> 784
# Dense: fully connected
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(25, activation='tanh', kernel_initializer=initializer, bias_initializer='zeros'),
    keras.layers.Dense(10, activation='sigmoid', kernel_initializer=initializer, bias_initializer='zeros')
    ])

opt = keras.optimizers.SGD(learning_rate=0.01)

model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])

history = model.fit(train_images,
                    train_labels,
                    validation_data=(test_images, test_labels),
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    verbose=2,
                    shuffle=True)
