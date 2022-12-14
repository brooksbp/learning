import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2
import numpy as np
import logging

tf.get_logger().setLevel(logging.ERROR)

EPOCHS = 500
BATCH_SIZE = 16

boston_housing = keras.datasets.boston_housing
(raw_x_train, y_train), (raw_x_test, y_test) = boston_housing.load_data()

# compute mean and stddev for each feature/variable/column of data
x_mean = np.mean(raw_x_train, axis=0)
x_stddev = np.std(raw_x_train, axis=0)

x_train = (raw_x_train - x_mean) / x_stddev
x_test = (raw_x_test - x_mean) / x_stddev

model = Sequential()
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.1), input_shape=[13]))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.1)))
model.add(Dropout(0.2))
model.add(Dense(1, activation='linear', kernel_regularizer=l2(0.1)))
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mean_absolute_error'])
model.summary()
history = model.fit(x_train,
                    y_train,
                    validation_data=(x_test, y_test),
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    verbose=2,
                    shuffle=True)

# after 500 epochs:
#
# baseline
# baseline + l2 regularization
# baseline + dropout
# baseline + l2 regularization + dropout
#
# loss: 0.6325 - mean_absolute_error: 0.5763 - val_loss: 13.0789 - val_mean_absolute_error: 2.5216
# loss: 8.7265 - mean_absolute_error: 1.5512 - val_loss: 19.0776 - val_mean_absolute_error: 2.4452
# loss: 8.4064 - mean_absolute_error: 2.1844 - val_loss: 14.9416 - val_mean_absolute_error: 2.4964
# loss: 13.8118 - mean_absolute_error: 2.2549 - val_loss: 20.6523 - val_mean_absolute_error: 2.7066

predictions = model.predict(x_test)

# print first 4 predictions
for i in range(0, 4):
    print('Prediction: ', predictions[i],
          ', true value:', y_test[i])
