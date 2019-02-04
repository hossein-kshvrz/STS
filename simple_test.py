import os

import numpy as np
from keras.callbacks import ModelCheckpoint

from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

batch_size = 128


def get_data():
    X = np.random.standard_normal(batch_size)
    Y = 3 * X + 10
    return X, Y


model = Sequential()
model.add(Dense(1, input_shape=(1, ), activation=None))
model.summary()
adam = Adam(lr=0.5)

x, y = get_data()
print(type(x))
x_t, y_t = get_data()


filepath = 'weights/weights.last.hdf5'

exists = os.path.isfile(filepath)

if exists:
    model.load_weights(filepath)

model.compile(loss='mse', optimizer=adam)

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, save_weights_only=True, mode='auto')
callbacks_list = [checkpoint]

history = model.fit(x, y, batch_size=batch_size, epochs=100, verbose=1, validation_data=None, callbacks=callbacks_list)

print(model.predict(x_t, verbose=1))
print(model.get_weights())

