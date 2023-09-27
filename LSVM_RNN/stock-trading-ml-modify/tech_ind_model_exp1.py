import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from keras import optimizers
import numpy as np
np.random.seed(4)
#from tensorflow import set_random_seed
#set_random_seed(4)
from util_exp1 import csv_to_dataset, history_points
import sys
import os.path

# dataset

ohlcv_histories, technical_indicators, y_normaliser, next_day_open_values, unscaled_y, next_day_open_values1, unscaled_y1, next_day_open_values2, unscaled_y2, next_day_open_values3, unscaled_y3, next_day_open_values4, unscaled_y4  = csv_to_dataset('MSFT_daily.csv')
test_split = 0.9
test_split = 0.9
n = int(ohlcv_histories.shape[0] * test_split)

# ohlcv_train = ohlcv_histories[:n]
# ohlcv_train=ohlcv_train[:,:,:2]
# print(ohlcv_train.shape)
# sys.exit()
ohlcv_train = ohlcv_histories[:n]
tech_ind_train = technical_indicators[:n]
y_train = next_day_open_values[:n]
y_train1 = next_day_open_values1[:n]
y_train2 = next_day_open_values2[:n]
y_train3 = next_day_open_values3[:n]
y_train4 = next_day_open_values4[:n]

ohlcv_test = ohlcv_histories[n:]
tech_ind_test = technical_indicators[n:]
y_test = next_day_open_values[n:]
y_test1 = next_day_open_values1[n:]
y_test2 = next_day_open_values2[n:]
y_test3 = next_day_open_values3[n:]
y_test4 = next_day_open_values4[n:]


unscaled_y_test = unscaled_y[n:]
unscaled_y_test1 = unscaled_y1[n:]
unscaled_y_test2 = unscaled_y2[n:]
unscaled_y_test3 = unscaled_y3[n:]
unscaled_y_test4 = unscaled_y4[n:]

print(ohlcv_train.shape)
print(ohlcv_test.shape)
print(tech_ind_train.shape)
print(tech_ind_test.shape)
print(y_train.shape)
print(y_test.shape)


# model architecture

# define two sets of inputs
lstm_input = Input(shape=(history_points, ohlcv_histories.shape[2]), name='lstm_input')
dense_input = Input(shape=(technical_indicators.shape[1],), name='tech_input')

# the first branch operates on the first input
x = LSTM(50)(lstm_input)
#x = Dropout(0.2)(x)
lstm_branch = Model(inputs=lstm_input, outputs=x)

# the second branch opreates on the second input
y = Dense(20, activation="relu")(dense_input)
#y = Dropout(0.2)(y)
technical_indicators_branch = Model(inputs=dense_input, outputs=y)

# combine the output of the two branches
combined = concatenate([lstm_branch.output, technical_indicators_branch.output])
z = Dense(64, activation="sigmoid")(combined)
z = Dense(1, activation="linear")(z)
# our model will accept the inputs of the two branches and
# then output a single value
model = Model(inputs=[lstm_branch.input, technical_indicators_branch.input], outputs=z)
model.summary()

# combined1 = concatenate([lstm_branch.output, model.output,technical_indicators_branch.output])
combined1 = concatenate([lstm_branch.output,technical_indicators_branch.output])
z1 = Dense(64, activation="sigmoid")(combined1)
z1 = Dense(1, activation="linear")(z1)
# our model will accept the inputs of the two branches and
# then output a single value
model1 = Model(inputs=[lstm_branch.input, technical_indicators_branch.input], outputs=z1)
model1.summary()

# combined2 = concatenate([lstm_branch.output, model.output, model1.output,technical_indicators_branch.output])
combined2 = concatenate([lstm_branch.output,technical_indicators_branch.output])
z2 = Dense(64, activation="sigmoid")(combined2)
z2 = Dense(1, activation="linear")(z2)
# our model will accept the inputs of the two branches and
# then output a single value
model2 = Model(inputs=[lstm_branch.input, technical_indicators_branch.input], outputs=z2)
model2.summary()

# combined3 = concatenate([lstm_branch.output, model.output, model1.output, model2.output, technical_indicators_branch.output])
combined3 = concatenate([lstm_branch.output, technical_indicators_branch.output])
z3 = Dense(64, activation="sigmoid")(combined3)
z3 = Dense(1, activation="linear")(z3)
# our model will accept the inputs of the two branches and
# then output a single value
model3 = Model(inputs=[lstm_branch.input, technical_indicators_branch.input], outputs=z3)
model3.summary()

modelN = Model(inputs=[lstm_branch.input, technical_indicators_branch.input], outputs=[model.output,model1.output,model2.output,model3.output])
modelN.summary()

# adam = optimizers.Adam(lr=0.0005)
adam = optimizers.Adam()
modelN.compile(loss=['mae', 'mae', 'mae', 'mae'], optimizer=adam)

history = modelN.fit([ohlcv_train, tech_ind_train], [y_train, y_train1, y_train2, y_train3], batch_size=1024, epochs=50, shuffle=True, validation_split=0.1)


# evaluation

y_test_predicted = model.predict([ohlcv_test, tech_ind_test])
y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)
y_predicted = model.predict([ohlcv_histories, technical_indicators])
y_predicted = y_normaliser.inverse_transform(y_predicted)
assert unscaled_y_test.shape == y_test_predicted.shape
real_mse = np.mean(np.square(unscaled_y_test - y_test_predicted))
scaled_mse = real_mse / (np.max(unscaled_y_test) - np.min(unscaled_y_test)) * 100
print(real_mse)
print(scaled_mse)


y_predicted,y_predicted1,y_predicted2,y_predicted3 = modelN.predict([ohlcv_histories, technical_indicators])
#y_predicted = model.predict([ohlcv_histories, technical_indicators])
#y_predicted1 = model1.predict([ohlcv_histories, technical_indicators])

y_predicted = y_normaliser.inverse_transform(y_predicted)
import matplotlib.pyplot as plt
plt.gcf().set_size_inches(22, 15, forward=True)
start = 0
end = -1
real = plt.plot(unscaled_y[start:end], label='real')
pred = plt.plot(y_predicted[start:end], label='predicted')
plt.legend(['Real', 'Predicted'])
plt.show()

y_predicted1 = y_normaliser.inverse_transform(y_predicted1)
plt.gcf().set_size_inches(22, 15, forward=True)
real = plt.plot(unscaled_y1[start:end], label='real1')
pred = plt.plot(y_predicted1[start:end], label='predicted1')
plt.legend(['Real1', 'Predicted1'])
plt.show()

y_predicted2 = y_normaliser.inverse_transform(y_predicted2)
plt.gcf().set_size_inches(22, 15, forward=True)
real = plt.plot(unscaled_y2[start:end], label='real2')
pred = plt.plot(y_predicted2[start:end], label='predicted2')
plt.legend(['Real2', 'Predicted2'])
plt.show()

y_predicted3 = y_normaliser.inverse_transform(y_predicted3)
plt.gcf().set_size_inches(22, 15, forward=True)
real = plt.plot(unscaled_y3[start:end], label='real3')
pred = plt.plot(y_predicted3[start:end], label='predicted3')
plt.legend(['Real3', 'Predicted3'])
plt.show()

plt.gcf().set_size_inches(22, 15, forward=True)
real = plt.plot(unscaled_y[start:end], label='real')
pred = plt.plot(y_predicted[start:end], label='predicted')
pred = plt.plot(y_predicted1[start:end], label='predicted1')
pred = plt.plot(y_predicted2[start:end], label='predicted2')
pred = plt.plot(y_predicted3[start:end], label='predicted3')
plt.legend(['Real', 'Predicted', 'Predicted1', 'Predicted2', 'Predicted3'])
plt.show()

plt.gcf().set_size_inches(22, 15, forward=True)
real = plt.plot(unscaled_y[start:end], label='real')
pred = plt.plot(unscaled_y1[start:end], label='real1')
pred = plt.plot(unscaled_y2[start:end], label='real2')
pred = plt.plot(unscaled_y3[start:end], label='real3')
plt.legend(['Real', 'Real1', 'Real2', 'Real3'])
plt.show()

from datetime import datetime
model.save(f'technical_model_N.h5')
