# import keras
# import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate, RNN, SimpleRNN,GRU
from keras import optimizers
import numpy as np
np.random.seed(4)
#from tensorflow import set_random_seed
#set_random_seed(4)
from util_N import csv_to_dataset, history_points
import sys

# import tensorflow as tf
import numpy as np

# dataset

ohlcv_histories, technical_indicators, next_day_open_values, unscaled_y, y_normaliser = csv_to_dataset('MSFT_daily.csv')
# sys.exit()
test_split = 0.9
n = int(ohlcv_histories.shape[0] * test_split)

ohlcv_train = ohlcv_histories[:n]
tech_ind_train = technical_indicators[:n]
y_train = next_day_open_values[:n]
# y_train = technical_indicators[:,0][:n]

ohlcv_test = ohlcv_histories[n:]
tech_ind_test = technical_indicators[n:]
y_test = next_day_open_values[n:]
# y_test = technical_indicators[:,0][n:]

unscaled_y_test = unscaled_y[n:]

# ohlcv_train = tf.convert_to_tensor(ohlcv_train)
# technical_indicators = tf.convert_to_tensor(technical_indicators)

print(ohlcv_train.shape)
print(ohlcv_test.shape)
print(tech_ind_train.shape)
print(tech_ind_test.shape)
print(y_train.shape)
print(y_test.shape)

# model architecture

# define two sets of inputs
lstm_input = Input(shape=(ohlcv_train.shape[1], ohlcv_train.shape[2]), name='lstm_input')
dense_input = Input(shape=(technical_indicators.shape[1],), name='tech_input')
# sys.exit()

# the first branch operates on the first input
x = LSTM(50, name='lstm_0')(lstm_input)
# x = SimpleRNN(50)(lstm_input)
# x = GRU(50)(lstm_input)
# x = keras.layers.RNN(
#     keras.layers.LSTMCell(50)
# )(lstm_input)



x = Dropout(0.2, name='lstm_dropout_0')(x)
lstm_branch = Model(inputs=lstm_input, outputs=x)

# the second branch opreates on the second input
y = Dense(20, name='tech_dense_0')(dense_input)
y = Activation("relu", name='tech_relu_0')(y)
y = Dropout(0.2, name='tech_dropout_0')(y)
technical_indicators_branch = Model(inputs=dense_input, outputs=y)

# combine the output of the two branches
combined = concatenate([lstm_branch.output, technical_indicators_branch.output], name='concatenate')

z = Dense(64, activation="sigmoid", name='dense_pooling')(combined)
# z = Dense(1, activation="linear", name='dense_out')(z)
z = Dense(1)(lstm_branch.output)#(z)

# our model will accept the inputs of the two branches and
# then output a single value
model = Model(inputs=[lstm_branch.input, technical_indicators_branch.input], outputs=z)
adam = optimizers.Adam(lr=0.0005)
model.compile(optimizer=adam, loss='mse')
model.fit(x=[ohlcv_train, tech_ind_train], y=y_train, batch_size=32, epochs=50, shuffle=True, validation_split=0.1)


# evaluation

y_test_predicted = model.predict([ohlcv_test, tech_ind_test])
y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)
y_predicted = model.predict([ohlcv_histories, technical_indicators])
y_predicted = y_normaliser.inverse_transform(y_predicted)

next_day_open = y_normaliser.inverse_transform(next_day_open_values)

assert unscaled_y_test.shape == y_test_predicted.shape
real_mse = np.mean(np.square(unscaled_y_test - y_test_predicted))
scaled_mse = real_mse / (np.max(unscaled_y_test) - np.min(unscaled_y_test)) * 100
print(real_mse)
print(scaled_mse)

import matplotlib.pyplot as plt

plt.gcf().set_size_inches(22, 15, forward=True)

start = 0
end = -1

# real = plt.plot(unscaled_y_test[start:end], label='real')
# pred = plt.plot(y_test_predicted[start:end], label='predicted')

real = plt.plot(unscaled_y[start:end], label='real')
pred = plt.plot(y_predicted[start:end], label='predicted')
pred = plt.plot(next_day_open[start:end], label='next_day_open')

plt.legend(['Real', 'Predicted','next_day_open'])

plt.show()


# real = plt.plot(technical_indicators[start:end,0], label='real')
# pred = plt.plot(y_predicted[start:end], label='predicted')

# plt.legend(['Real', 'Predicted'])

# plt.show()


plt.gcf().set_size_inches(22, 15, forward=True)

start = 0
end = -1

real = plt.plot(unscaled_y_test[start:end], label='real')
pred = plt.plot(y_test_predicted[start:end], label='predicted')

plt.legend(['Real', 'Predicted'])

plt.show()

# real = plt.plot(y_test[start:end], label='real')
# pred = plt.plot(y_test_predicted[start:end], label='predicted')

# plt.legend(['Real', 'Predicted'])

# plt.show()

# plt.gcf().set_size_inches(22, 15, forward=True)

# start = 0
# end = -1

# # real = plt.plot(unscaled_y_test[start:end], label='real')
# # pred = plt.plot(y_test_predicted[start:end], label='predicted')

# # plt.legend(['Real', 'Predicted'])

# # plt.show()

# real = plt.plot(technical_indicators[start:end,0], label='real')
# pred = plt.plot(y_predicted[start:end], label='predicted')

# plt.legend(['Real', 'Predicted'])

# plt.show()


from datetime import datetime
model.save(f'technical_model_N.h5')
