import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate, RNN, SimpleRNN,GRU,Embedding
from keras import optimizers
import numpy as np
np.random.seed(4)
#from tensorflow import set_random_seed
#set_random_seed(4)
from util_N_ind_gru import csv_to_dataset, history_points
import sys

# dataset

ohlcv_histories, technical_indicators, next_day_open_values, unscaled_y, y_normaliser = csv_to_dataset('MSFT_daily.csv')
# sys.exit()
test_split = 0.9
n = int(ohlcv_histories.shape[0] * test_split)

ohlcv_train = ohlcv_histories[:n]
tech_ind_train = technical_indicators[:n]
y_train = next_day_open_values[:n]
# y_train = technical_indicators[:,0][:n]
# y_train1 = technical_indicators[:,1][:n]

ohlcv_test = ohlcv_histories[n:]
tech_ind_test = technical_indicators[n:]
y_test = next_day_open_values[n:]
# y_test = technical_indicators[:,0][n:]
# y_test1 = technical_indicators[:,1][n:]

unscaled_y_test = unscaled_y[n:]

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

print(lstm_input.shape)
print(dense_input.shape)

class MyModel(tf.keras.Model):
  def __init__(self, vocab_size, rnn_units):
    super().__init__(self)
    self.gru = tf.keras.layers.GRU(rnn_units,
                                   return_sequences=True,
                                   return_state=True)
    self.dense = tf.keras.layers.Dense(vocab_size)

  def call(self, inputs, states=None, return_state=False, training=False):
    x = inputs
    if states is None:
      states = self.gru.get_initial_state(x)
    x, states = self.gru(x, initial_state=states, training=training)
    x = self.dense(x, training=training)

    if return_state:
      return x, states
    else:
      return x

model = MyModel(
    vocab_size=1,
    rnn_units=lstm_input.shape[1])



# # y_test_predicted = model.predict([ohlcv_test, tech_ind_test])
# states=None
# inp=np.zeros((1,ohlcv_test.shape[1],ohlcv_test.shape[2]))
# ohlcv_test_predicted = np.zeros((ohlcv_test.shape[0],ohlcv_test.shape[1],1))
# for r1 in range(2):
#     for r in range(ohlcv_test.shape[0]):
#         inp[0]=ohlcv_test[r]
#         # print(inp.shape)
# # y_test_predicted, states = model(inputs=ohlcv_test, states=states,return_state=True)
#         y_test_predicted, states = model(inputs=inp, states=states,return_state=True)
#         y_test_predicted=y_test_predicted.numpy()
#         # print(y_test_predicted.shape)
#         # y_test_predicted=y_test_predicted.reshape((y_test_predicted.shape[0], y_test_predicted.shape[1]))
#         # y_test_predicted=y_test_predicted.reshape((y_test_predicted.shape[0]*y_test_predicted.shape[1],-1 ))
#         # y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)
#         # print(y_test_predicted.shape)
#         # print(y_test_predicted[0])
#         ohlcv_test_predicted[r]=y_test_predicted

# print(ohlcv_test_predicted.shape)
# y_test_predicted = ohlcv_test_predicted
# y_test_predicted=y_test_predicted.reshape((y_test_predicted.shape[0], y_test_predicted.shape[1]))
# y_test_predicted=y_test_predicted.reshape((y_test_predicted.shape[0]*y_test_predicted.shape[1],-1 ))
# y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)
# print(y_test_predicted.shape)
# sys.exit()
# model.summary()

adam = optimizers.Adam(lr=0.0005)
# model.compile(optimizer=adam, loss=['mse'])
model.compile(optimizer='adam', loss="mse")
# model.summary()
# history = model.fit(x=[ohlcv_train, tech_ind_train], y=[y_train], batch_size=32, epochs=50, shuffle=True, validation_split=0.1)
history = model.fit(x=ohlcv_train, y=y_train, epochs=50)
# sys.exit()
# evaluation
states=None
y_test_predicted, states = model(inputs=ohlcv_test, states=states,return_state=True)
y_test_predicted=y_test_predicted.numpy()
y_test_predicted.reshape((y_test_predicted.shape[0], y_test_predicted.shape[1]))
y_test_predicted=y_test_predicted.reshape((y_test_predicted.shape[0]*y_test_predicted.shape[1],-1 ))
# y_test_predicted=np.ravel(y_test_predicted)
y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)

# states=None
# y_predicted, states = model(inputs=ohlcv_histories, states=states,return_state=True)
# y_predicted=y_predicted.numpy()
# y_predicted.reshape((y_predicted.shape[0], y_predicted.shape[1]))
# y_predicted=y_predicted.reshape((y_predicted.shape[0]*y_predicted.shape[1],-1 ))
# # y_predicted=np.ravel(y_predicted)
# y_predicted = y_normaliser.inverse_transform(y_predicted)

states=None
inp=np.zeros((1,ohlcv_histories.shape[1],ohlcv_histories.shape[2]))
ohlcv_histories_predicted = np.zeros((ohlcv_histories.shape[0],ohlcv_histories.shape[1],1))
for r1 in range(2):
    for r in range(ohlcv_histories.shape[0]):
        inp[0]=ohlcv_histories[r]
        # print(inp.shape)
# y_predicted, states = model(inputs=ohlcv_histories, states=states,return_state=True)
        y_predicted, states = model(inputs=inp, states=states,return_state=True)
        y_predicted=y_predicted.numpy()
        # print(y_predicted.shape)
        # y_predicted=y_predicted.reshape((y_predicted.shape[0], y_predicted.shape[1]))
        # y_predicted=y_predicted.reshape((y_predicted.shape[0]*y_predicted.shape[1],-1 ))
        # y_predicted = y_normaliser.inverse_transform(y_predicted)
        # print(y_predicted.shape)
        # print(y_predicted[0])
        ohlcv_histories_predicted[r]=y_predicted

y_predicted = ohlcv_histories_predicted
y_predicted=y_predicted.reshape((y_predicted.shape[0], y_predicted.shape[1]))
y_predicted=y_predicted.reshape((y_predicted.shape[0]*y_predicted.shape[1],-1 ))
y_predicted = y_normaliser.inverse_transform(y_predicted)


next_day_open_values.reshape((next_day_open_values.shape[0], next_day_open_values.shape[1]))
next_day_open_values=next_day_open_values.reshape((next_day_open_values.shape[0]*next_day_open_values.shape[1],-1 ))
# next_day_open_values=np.ravel(next_day_open_values)
next_day_open = y_normaliser.inverse_transform(next_day_open_values)

unscaled_y = unscaled_y.reshape((unscaled_y.shape[0], unscaled_y.shape[1]))
unscaled_y=unscaled_y.reshape((unscaled_y.shape[0]*unscaled_y.shape[1],-1 ))
# unscaled_y=np.ravel(unscaled_y)

unscaled_y_test = unscaled_y_test.reshape((unscaled_y_test.shape[0], unscaled_y_test.shape[1]))
unscaled_y_test=unscaled_y_test.reshape((unscaled_y_test.shape[0]*unscaled_y_test.shape[1],-1 ))


import matplotlib.pyplot as plt

plt.gcf().set_size_inches(22, 15, forward=True)

start = 0
end = -1

real = plt.plot(unscaled_y[start:end], label='real')
pred = plt.plot(y_predicted[start:end], label='predicted')
pred = plt.plot(next_day_open[start:end], label='next_day_open')

plt.legend(['Real', 'Predicted','next_day_open'])

plt.show()


plt.gcf().set_size_inches(22, 15, forward=True)

start = 0
end = -1

plt.gcf().set_size_inches(22, 15, forward=True)

start = 0
end = -1

real = plt.plot(unscaled_y_test[start:end], label='real')
pred = plt.plot(y_test_predicted[start:end], label='predicted')

plt.legend(['Real', 'Predicted'])

plt.show()

# from datetime import datetime
# model.save(f'technical_model_N.h5')

# sys.exit()

x, states = GRU(lstm_input.shape[1],return_sequences=True,return_state=True)(lstm_input)

lstm_branch = Model(inputs=lstm_input, outputs=x)

y = Dense(20, name='tech_dense_0')(dense_input)
y = Activation("relu", name='tech_relu_0')(y)
y = Dropout(0.2, name='tech_dropout_0')(y)

technical_indicators_branch = Model(inputs=dense_input, outputs=y)
z = Dense(1)(lstm_branch.output)#(z)
model = Model(inputs=[lstm_branch.input, technical_indicators_branch.input], outputs=[z])

y_test_predicted = model.predict([ohlcv_test, tech_ind_test])
print(y_test_predicted.shape)
y_test_predicted=y_test_predicted.reshape((y_test_predicted.shape[0], y_test_predicted.shape[1]))
y_test_predicted=y_test_predicted.reshape((y_test_predicted.shape[0]*y_test_predicted.shape[1],-1 ))
y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)
print(y_test_predicted.shape)
# print(y_test_predicted[0])
# sys.exit()

adam = optimizers.Adam(lr=0.0005)
model.compile(optimizer=adam, loss=['mse'])
model.summary()
history = model.fit(x=[ohlcv_train, tech_ind_train], y=[y_train], epochs=50)

y_test_predicted = model.predict([ohlcv_test, tech_ind_test])
y_test_predicted.reshape((y_test_predicted.shape[0], y_test_predicted.shape[1]))
y_test_predicted=y_test_predicted.reshape((y_test_predicted.shape[0]*y_test_predicted.shape[1],-1 ))
# y_test_predicted=np.ravel(y_test_predicted)
y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)
y_predicted = model.predict([ohlcv_histories, technical_indicators])
y_predicted.reshape((y_predicted.shape[0], y_predicted.shape[1]))
y_predicted=y_predicted.reshape((y_predicted.shape[0]*y_predicted.shape[1],-1 ))
# y_predicted=np.ravel(y_predicted)
y_predicted = y_normaliser.inverse_transform(y_predicted)

next_day_open_values.reshape((next_day_open_values.shape[0], next_day_open_values.shape[1]))
next_day_open_values=next_day_open_values.reshape((next_day_open_values.shape[0]*next_day_open_values.shape[1],-1 ))
# next_day_open_values=np.ravel(next_day_open_values)
next_day_open = y_normaliser.inverse_transform(next_day_open_values)

unscaled_y = unscaled_y.reshape((unscaled_y.shape[0], unscaled_y.shape[1]))
unscaled_y=unscaled_y.reshape((unscaled_y.shape[0]*unscaled_y.shape[1],-1 ))
# unscaled_y=np.ravel(unscaled_y)

unscaled_y_test = unscaled_y_test.reshape((unscaled_y_test.shape[0], unscaled_y_test.shape[1]))
unscaled_y_test=unscaled_y_test.reshape((unscaled_y_test.shape[0]*unscaled_y_test.shape[1],-1 ))

import matplotlib.pyplot as plt

plt.gcf().set_size_inches(22, 15, forward=True)

start = 0
end = -1

real = plt.plot(unscaled_y[start:end], label='real')
pred = plt.plot(y_predicted[start:end], label='predicted')
pred = plt.plot(next_day_open[start:end], label='next_day_open')

plt.legend(['Real', 'Predicted','next_day_open'])

plt.show()


plt.gcf().set_size_inches(22, 15, forward=True)

start = 0
end = -1

plt.gcf().set_size_inches(22, 15, forward=True)

start = 0
end = -1

real = plt.plot(unscaled_y_test[start:end], label='real')
pred = plt.plot(y_test_predicted[start:end], label='predicted')

plt.legend(['Real', 'Predicted'])

plt.show()

from datetime import datetime
model.save(f'technical_model_N.h5')

sys.exit()
