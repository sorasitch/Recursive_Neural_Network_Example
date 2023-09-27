import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate, Reshape
from keras import optimizers
import numpy as np
np.random.seed(4)
#from tensorflow import set_random_seed
#set_random_seed(4)
from util_exp1 import csv_to_dataset, history_points
import sys
import os.path
import matplotlib.pyplot as plt
# dataset

ohlcv_histories, technical_indicators, y_normaliser, next_day_open_values, unscaled_y, next_day_open_values1, unscaled_y1, next_day_open_values2, unscaled_y2, next_day_open_values3, unscaled_y3, next_day_open_values4, unscaled_y4  = csv_to_dataset('F:/DRIVE_OWN/DeepLearning/example/LSVM/stock-trading-ml-master/MSFT_daily.csv')
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
y_input = Input(shape=(history_points, 1), name='y_input')

# the first branch operates on the first input
x = LSTM(50)(lstm_input)
#x = Dropout(0.2)(x)
lstm_branch = Model(inputs=lstm_input, outputs=x)

# the second branch opreates on the second input
# a = Reshape((1,dense_input.shape[1]))(dense_input)
# y = LSTM(20)(a)
# y = Dense(20, activation="relu")(y)
y = Dense(20, activation="relu")(dense_input)
#y = Dropout(0.2)(y)
technical_indicators_branch = Model(inputs=dense_input, outputs=y)

x = LSTM(50)(lstm_input)
#x = Dropout(0.2)(x)
lstm_branch = Model(inputs=lstm_input, outputs=x)

# the 3rd branch opreates on the 3rd input
v = LSTM(50)(y_input)
y_branch = Model(inputs=y_input, outputs=v)


# the first branch operates on the first input
x1 = LSTM(50)(lstm_input)
#x1 = Dropout(0.2)(x1)
lstm_branch1 = Model(inputs=lstm_input, outputs=x1)

# the second branch opreates on the second input
# a1 = Reshape((1,dense_input.shape[1]))(dense_input)
# y1 = LSTM(20)(a1)
# y1 = Dense(20, activation="relu")(y1)
y1 = Dense(20, activation="relu")(dense_input)
#y1 = Dropout(0.2)(y1)
technical_indicators_branch1 = Model(inputs=dense_input, outputs=y1)

# the 3rd branch opreates on the 3rd input
v1 = LSTM(50)(y_input)
y_branch1 = Model(inputs=y_input, outputs=v1)


# the first branch operates on the first input
x2 = LSTM(50)(lstm_input)
#x2 = Dropout(0.2)(x2)
lstm_branch2 = Model(inputs=lstm_input, outputs=x2)

# the second branch opreates on the second input
# a2 = Reshape((1,dense_input.shape[1]))(dense_input)
# y2 = LSTM(20)(a2)
# y2 = Dense(20, activation="relu")(y2)
y2 = Dense(20, activation="relu")(dense_input)
#y2 = Dropout(0.2)(y2)
technical_indicators_branch2 = Model(inputs=dense_input, outputs=y2)

# the 3rd branch opreates on the 3rd input
v2 = LSTM(50)(y_input)
y_branch2 = Model(inputs=y_input, outputs=v2)


# the first branch operates on the first input
x3 = LSTM(50)(lstm_input)
#x3 = Dropout(0.2)(x3)
lstm_branch3 = Model(inputs=lstm_input, outputs=x3)

# the second branch opreates on the second input
# a3 = Reshape((1,dense_input.shape[1]))(dense_input)
# y3 = LSTM(20)(a3)
# y3 = Dense(20, activation="relu")(y3)
y3 = Dense(20, activation="relu")(dense_input)
#y3 = Dropout(0.2)(y3)
technical_indicators_branch3 = Model(inputs=dense_input, outputs=y3)

# the 3rd branch opreates on the 3rd input
v3 = LSTM(50)(y_input)
y_branch3 = Model(inputs=y_input, outputs=v3)


# combine the output of the two branches
# combined = concatenate([lstm_branch.output, technical_indicators_branch.output])
combined = concatenate([lstm_branch.output, technical_indicators_branch.output, y_branch.output])
z = Dense(64, activation="sigmoid")(combined)
z = Dense(1, activation="linear")(z)
# our model will accept the inputs of the two branches and
# then output a single value
# model = Model(inputs=[lstm_branch.input, technical_indicators_branch.input], outputs=z)
model = Model(inputs=[lstm_branch.input, technical_indicators_branch.input, y_branch.input], outputs=z)
model.summary()

# combined1 = concatenate([lstm_branch1.output, model.output,technical_indicators_branch1.output])
# combined1 = concatenate([lstm_branch1.output,technical_indicators_branch1.output])
combined1 = concatenate([lstm_branch1.output,technical_indicators_branch1.output, y_branch1.output])
z1 = Dense(64, activation="sigmoid")(combined1)
z1 = Dense(1, activation="linear")(z1)
# our model will accept the inputs of the two branches and
# then output a single value
# model1 = Model(inputs=[lstm_branch1.input, technical_indicators_branch1.input], outputs=z1)
model1 = Model(inputs=[lstm_branch1.input, technical_indicators_branch1.input, y_branch1.input], outputs=z1)
model1.summary()

# combined2 = concatenate([lstm_branch2.output, model.output, model1.output,technical_indicators_branch2.output])
# combined2 = concatenate([lstm_branch2.output,technical_indicators_branch2.output])
combined2 = concatenate([lstm_branch2.output,technical_indicators_branch2.output, y_branch2.output])
z2 = Dense(64, activation="sigmoid")(combined2)
z2 = Dense(1, activation="linear")(z2)
# our model will accept the inputs of the two branches and
# then output a single value
# model2 = Model(inputs=[lstm_branch2.input, technical_indicators_branch2.input], outputs=z2)
model2 = Model(inputs=[lstm_branch2.input, technical_indicators_branch2.input, y_branch2.input], outputs=z2)
model2.summary()

# combined3 = concatenate([lstm_branch3.output, model.output, model1.output, model2.output, technical_indicators_branch3.output])
# combined3 = concatenate([lstm_branch3.output, technical_indicators_branch3.output])
combined3 = concatenate([lstm_branch3.output, technical_indicators_branch3.output, y_branch3.output])
z3 = Dense(64, activation="sigmoid")(combined3)
z3 = Dense(1, activation="linear")(z3)
# our model will accept the inputs of the two branches and
# then output a single value
# model3 = Model(inputs=[lstm_branch3.input, technical_indicators_branch3.input], outputs=z3)
model3 = Model(inputs=[lstm_branch3.input, technical_indicators_branch3.input, y_branch3.input], outputs=z3)
model3.summary()

# model

# modelN = Model(inputs=[lstm_branch.input, technical_indicators_branch.input], outputs=[model.output])
modelN = Model(inputs=[lstm_branch.input, technical_indicators_branch.input, y_branch.input], outputs=[model.output])
modelN.summary()

# modelN1 = Model(inputs=[lstm_branch1.input, technical_indicators_branch1.input], outputs=[model1.output])
modelN1 = Model(inputs=[lstm_branch1.input, technical_indicators_branch1.input, y_branch1.input], outputs=[model1.output])
modelN1.summary()

# modelN2 = Model(inputs=[lstm_branch2.input, technical_indicators_branch2.input], outputs=[model2.output])
modelN2 = Model(inputs=[lstm_branch2.input, technical_indicators_branch2.input, y_branch2.input], outputs=[model2.output])
modelN2.summary()

# modelN3 = Model(inputs=[lstm_branch3.input, technical_indicators_branch3.input], outputs=[model3.output])
modelN3 = Model(inputs=[lstm_branch3.input, technical_indicators_branch3.input, y_branch3.input], outputs=[model3.output])
modelN3.summary()


# adam = optimizers.Adam(lr=0.0005)
#adam = optimizers.Adam()
#adam1 = optimizers.Adam()
#adam2 = optimizers.Adam()
#adam3 = optimizers.Adam()


adam = tf.keras.optimizers.Adam()
adam1 = tf.keras.optimizers.Adam()
adam2 = tf.keras.optimizers.Adam()
adam3 = tf.keras.optimizers.Adam()


modelN.compile(loss=['mae'], optimizer=adam)
modelN1.compile(loss=['mae'], optimizer=adam)
modelN2.compile(loss=['mae'], optimizer=adam)
modelN3.compile(loss=['mae'], optimizer=adam)

# history = modelN.fit([ohlcv_train, tech_ind_train], [y_train], batch_size=1024, epochs=100, shuffle=True, validation_split=0.1)
# history1 = modelN1.fit([ohlcv_train, tech_ind_train], [y_train1], batch_size=1024, epochs=100, shuffle=True, validation_split=0.1)
# history2 = modelN2.fit([ohlcv_train, tech_ind_train], [y_train2], batch_size=1024, epochs=100, shuffle=True, validation_split=0.1)
# history3 = modelN3.fit([ohlcv_train, tech_ind_train], [y_train3], batch_size=1024, epochs=100, shuffle=True, validation_split=0.1)
history = modelN.fit([ohlcv_train, tech_ind_train, ohlcv_train[:,:,0]], [y_train], batch_size=1024, epochs=100, shuffle=True, validation_split=0.1)
history1 = modelN1.fit([ohlcv_train, tech_ind_train, ohlcv_train[:,:,0]], [y_train1], batch_size=1024, epochs=100, shuffle=True, validation_split=0.1)
history2 = modelN2.fit([ohlcv_train, tech_ind_train, ohlcv_train[:,:,0]], [y_train2], batch_size=1024, epochs=100, shuffle=True, validation_split=0.1)
history3 = modelN3.fit([ohlcv_train, tech_ind_train, ohlcv_train[:,:,0]], [y_train3], batch_size=1024, epochs=100, shuffle=True, validation_split=0.1)


# evaluation

# y_test_predicted = model.predict([ohlcv_test, tech_ind_test])
y_test_predicted = model.predict([ohlcv_test, tech_ind_test, ohlcv_test[:,:,0]])
y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)
# y_predicted = model.predict([ohlcv_histories, technical_indicators])
y_predicted = model.predict([ohlcv_histories, technical_indicators, ohlcv_histories[:,:,0]])
y_predicted = y_normaliser.inverse_transform(y_predicted)
assert unscaled_y_test.shape == y_test_predicted.shape
real_mse = np.mean(np.square(unscaled_y_test - y_test_predicted))
scaled_mse = real_mse / (np.max(unscaled_y_test) - np.min(unscaled_y_test)) * 100
print(real_mse)
print(scaled_mse)


# y_predicted = modelN.predict([ohlcv_histories, technical_indicators])
# y_predicted1 = modelN1.predict([ohlcv_histories, technical_indicators])
# y_predicted2 = modelN2.predict([ohlcv_histories, technical_indicators])
# y_predicted3 = modelN3.predict([ohlcv_histories, technical_indicators])
y_predicted = modelN.predict([ohlcv_histories, technical_indicators, ohlcv_histories[:,:,0]])
y_predicted1 = modelN1.predict([ohlcv_histories, technical_indicators, ohlcv_histories[:,:,0]])
y_predicted2 = modelN2.predict([ohlcv_histories, technical_indicators, ohlcv_histories[:,:,0]])
y_predicted3 = modelN3.predict([ohlcv_histories, technical_indicators, ohlcv_histories[:,:,0]])

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
real = plt.plot(unscaled_y1[start:end], label='real')
pred = plt.plot(y_predicted[start:end], label='predicted')
pred = plt.plot(y_predicted1[start:end], label='predicted1')
pred = plt.plot(y_predicted2[start:end], label='predicted2')
pred = plt.plot(y_predicted3[start:end], label='predicted3')
plt.legend(['Real','Real1', 'Predicted', 'Predicted1', 'Predicted2', 'Predicted3'])
plt.show()

plt.gcf().set_size_inches(22, 15, forward=True)
real = plt.plot(unscaled_y[start:end], label='real')
pred = plt.plot(unscaled_y1[start:end], label='real1')
pred = plt.plot(unscaled_y2[start:end], label='real2')
pred = plt.plot(unscaled_y3[start:end], label='real3')
plt.legend(['Real', 'Real1', 'Real2', 'Real3'])
plt.show()

plt.gcf().set_size_inches(22, 15, forward=True)
real = plt.plot(y_normaliser.inverse_transform(np.expand_dims(technical_indicators[start:end,0],-1)), label='technical_indicators')
real = plt.plot(y_normaliser.inverse_transform(np.expand_dims(technical_indicators[start:end,1],-1)), label='technical_indicators1')
pred = plt.plot(y_predicted[start:end], label='predicted')
pred = plt.plot(y_predicted1[start:end], label='predicted1')
pred = plt.plot(y_predicted2[start:end], label='predicted2')
pred = plt.plot(y_predicted3[start:end], label='predicted3')
plt.legend(['technical_indicators', 'technical_indicators1','Predicted', 'Predicted1', 'Predicted2', 'Predicted3'])
plt.show()

plt.gcf().set_size_inches(22, 15, forward=True)
real = plt.plot(ohlcv_histories[:,0,:], label='ohlcv_histories')
real = plt.plot(next_day_open_values[:], label='real')
plt.show()

from datetime import datetime
model.save(f'technical_model_N.h5')
