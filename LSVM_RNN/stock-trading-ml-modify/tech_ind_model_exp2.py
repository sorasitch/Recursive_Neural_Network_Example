import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate, add, Reshape, TimeDistributed, Reshape
from keras import optimizers , Sequential
from  keras.backend import expand_dims
import numpy as np
np.random.seed(4)
#from tensorflow import set_random_seed
#set_random_seed(4)
from util_exp2 import csv_to_dataset, history_points
import sys
import os.path

# dataset

ohlcv_histories, technical_indicators, y_normaliser, y_normaliser_volume, next_day_open_values, unscaled_y, next_day_open_values1, unscaled_y1, next_day_open_values2, unscaled_y2, next_day_open_values3, unscaled_y3, next_day_open_values4, unscaled_y4  = csv_to_dataset('MSFT_daily.csv')
test_split = 0.9
test_split = 0.9
n = int(ohlcv_histories.shape[0] * test_split)

# ohlcv_train = ohlcv_histories[:n]
# ohlcv_train=ohlcv_train[:,:,:2]
# print(ohlcv_train.shape)
# sys.exit()
ohlcv_train = ohlcv_histories[:n]
tech_ind_train = technical_indicators[:n]
y_train = next_day_open_values[:n,0]
y_train=np.expand_dims(y_train, -1)
y_train1 = next_day_open_values1[:n,0]
y_train1=np.expand_dims(y_train1, -1)
y_train2 = next_day_open_values2[:n,0]
y_train2=np.expand_dims(y_train2, -1)
y_train3 = next_day_open_values3[:n,0]
y_train3=np.expand_dims(y_train3, -1)
y_train4 = next_day_open_values4[:n,0]
y_train4=np.expand_dims(y_train4, -1)
y_train_hi = next_day_open_values[:n,1]
y_train_hi=np.expand_dims(y_train_hi, -1)
y_train1_hi = next_day_open_values1[:n,1]
y_train1_hi=np.expand_dims(y_train1_hi, -1)
y_train2_hi = next_day_open_values2[:n,1]
y_train2_hi=np.expand_dims(y_train2_hi, -1)
y_train3_hi = next_day_open_values3[:n,1]
y_train3_hi=np.expand_dims(y_train3_hi, -1)
y_train4_hi = next_day_open_values4[:n,1]
y_train4_hi=np.expand_dims(y_train4_hi, -1)
y_train_lo = next_day_open_values[:n,2]
y_train_lo=np.expand_dims(y_train_lo, -1)
y_train1_lo = next_day_open_values1[:n,2]
y_train1_lo=np.expand_dims(y_train1_lo, -1)
y_train2_lo = next_day_open_values2[:n,2]
y_train2_lo=np.expand_dims(y_train2_lo, -1)
y_train3_lo = next_day_open_values3[:n,2]
y_train3_lo=np.expand_dims(y_train3_lo, -1)
y_train4_lo = next_day_open_values4[:n,2]
y_train4_lo=np.expand_dims(y_train4_lo, -1)
y_train_cl = next_day_open_values[:n,3]
y_train_cl=np.expand_dims(y_train_cl, -1)
y_train1_cl = next_day_open_values1[:n,3]
y_train1_cl=np.expand_dims(y_train1_cl, -1)
y_train2_cl = next_day_open_values2[:n,3]
y_train2_cl=np.expand_dims(y_train2_cl, -1)
y_train3_cl = next_day_open_values3[:n,3]
y_train3_cl=np.expand_dims(y_train3_cl, -1)
y_train4_cl = next_day_open_values4[:n,3]
y_train4_cl=np.expand_dims(y_train4_cl, -1)
y_train_va = next_day_open_values[:n,4]
y_train_va=np.expand_dims(y_train_va, -1)
y_train1_va = next_day_open_values1[:n,4]
y_train1_va=np.expand_dims(y_train1_va, -1)
y_train2_va = next_day_open_values2[:n,4]
y_train2_va=np.expand_dims(y_train2_va, -1)
y_train3_va = next_day_open_values3[:n,4]
y_train3_va=np.expand_dims(y_train3_va, -1)
y_train4_va = next_day_open_values4[:n,4]
y_train4_va=np.expand_dims(y_train4_va, -1)
# print(y_train4.shape)

ohlcv_test = ohlcv_histories[n:]
tech_ind_test = technical_indicators[n:]
y_test = next_day_open_values[n:,0]
y_test=np.expand_dims(y_test, -1)
y_test1 = next_day_open_values1[n:,0]
y_test1=np.expand_dims(y_test1, -1)
y_test2 = next_day_open_values2[n:,0]
y_test2=np.expand_dims(y_test2, -1)
y_test3 = next_day_open_values3[n:,0]
y_test3=np.expand_dims(y_test3, -1)
y_test4 = next_day_open_values4[n:,0]
y_test4=np.expand_dims(y_test4, -1)
y_test_hi = next_day_open_values[n:,1]
y_test_hi=np.expand_dims(y_test_hi, -1)
y_test1_hi = next_day_open_values1[n:,1]
y_test1_hi=np.expand_dims(y_test1_hi, -1)
y_test2_hi = next_day_open_values2[n:,1]
y_test2_hi=np.expand_dims(y_test2_hi, -1)
y_test3_hi = next_day_open_values3[n:,1]
y_test3_hi=np.expand_dims(y_test3_hi, -1)
y_test4_hi = next_day_open_values4[n:,1]
y_test4_hi=np.expand_dims(y_test4_hi, -1)
y_test_lo = next_day_open_values[n:,2]
y_test_lo=np.expand_dims(y_test_lo, -1)
y_test1_lo = next_day_open_values1[n:,2]
y_test1_lo=np.expand_dims(y_test1_lo, -1)
y_test2_lo = next_day_open_values2[n:,2]
y_test2_lo=np.expand_dims(y_test2_lo, -1)
y_test3_lo = next_day_open_values3[n:,2]
y_test3_lo=np.expand_dims(y_test3_lo, -1)
y_test4_lo = next_day_open_values4[n:,2]
y_test4_lo=np.expand_dims(y_test4_lo, -1)
y_test_cl = next_day_open_values[n:,3]
y_test_cl=np.expand_dims(y_test_cl, -1)
y_test1_cl = next_day_open_values1[n:,3]
y_test1_cl=np.expand_dims(y_test1_cl, -1)
y_test2_cl = next_day_open_values2[n:,3]
y_test2_cl=np.expand_dims(y_test2_cl, -1)
y_test3_cl = next_day_open_values3[n:,3]
y_test3_cl=np.expand_dims(y_test3_cl, -1)
y_test4_cl = next_day_open_values4[n:,3]
y_test4_cl=np.expand_dims(y_test4_cl, -1)
y_test_va = next_day_open_values[n:,4]
y_test_va=np.expand_dims(y_test_va, -1)
y_test1_va = next_day_open_values1[n:,4]
y_test1_va=np.expand_dims(y_test1_va, -1)
y_test2_va = next_day_open_values2[n:,4]
y_test2_va=np.expand_dims(y_test2_va, -1)
y_test3_va = next_day_open_values3[n:,4]
y_test3_va=np.expand_dims(y_test3_va, -1)
y_test4_va = next_day_open_values4[n:,4]
y_test4_va=np.expand_dims(y_test4_va, -1)

unscaled_y_test = unscaled_y[n:,0]
unscaled_y_test=np.expand_dims(unscaled_y_test, -1)
unscaled_y_test1 = unscaled_y1[n:,0]
unscaled_y_test1=np.expand_dims(unscaled_y_test1, -1)
unscaled_y_test2 = unscaled_y2[n:,0]
unscaled_y_test2=np.expand_dims(unscaled_y_test2, -1)
unscaled_y_test3 = unscaled_y3[n:,0]
unscaled_y_test3=np.expand_dims(unscaled_y_test3, -1)
unscaled_y_test4 = unscaled_y4[n:,0]
unscaled_y_test4=np.expand_dims(unscaled_y_test4, -1)
unscaled_y_test_hi = unscaled_y[n:,1]
unscaled_y_test_hi=np.expand_dims(unscaled_y_test_hi, -1)
unscaled_y_test1_hi = unscaled_y1[n:,1]
unscaled_y_test1_hi=np.expand_dims(unscaled_y_test1_hi, -1)
unscaled_y_test2_hi = unscaled_y2[n:,1]
unscaled_y_test2_hi=np.expand_dims(unscaled_y_test2_hi, -1)
unscaled_y_test3_hi = unscaled_y3[n:,1]
unscaled_y_test3_hi=np.expand_dims(unscaled_y_test3_hi, -1)
unscaled_y_test4_hi = unscaled_y4[n:,1]
unscaled_y_test4_hi=np.expand_dims(unscaled_y_test4_hi, -1)
unscaled_y_test_lo = unscaled_y[n:,2]
unscaled_y_test_lo=np.expand_dims(unscaled_y_test_lo, -1)
unscaled_y_test1_lo = unscaled_y1[n:,2]
unscaled_y_test1_lo=np.expand_dims(unscaled_y_test1_lo, -1)
unscaled_y_test2_lo = unscaled_y2[n:,2]
unscaled_y_test2_lo=np.expand_dims(unscaled_y_test2_lo, -1)
unscaled_y_test3_lo = unscaled_y3[n:,2]
unscaled_y_test3_lo=np.expand_dims(unscaled_y_test3_lo, -1)
unscaled_y_test4_lo = unscaled_y4[n:,2]
unscaled_y_test4=np.expand_dims(unscaled_y_test4_lo, -1)
unscaled_y_test_cl = unscaled_y[n:,3]
unscaled_y_test_cl=np.expand_dims(unscaled_y_test_cl, -1)
unscaled_y_test1_cl = unscaled_y1[n:,3]
unscaled_y_test1_cl=np.expand_dims(unscaled_y_test1_cl, -1)
unscaled_y_test2_cl = unscaled_y2[n:,3]
unscaled_y_test2_cl=np.expand_dims(unscaled_y_test2_cl, -1)
unscaled_y_test3_cl = unscaled_y3[n:,3]
unscaled_y_test3_cl=np.expand_dims(unscaled_y_test3_cl, -1)
unscaled_y_test4_cl = unscaled_y4[n:,3]
unscaled_y_test4_cl=np.expand_dims(unscaled_y_test4_cl, -1)
unscaled_y_test_va = unscaled_y[n:,4]
unscaled_y_test_va=np.expand_dims(unscaled_y_test_va, -1)
unscaled_y_test1_va = unscaled_y1[n:,4]
unscaled_y_test1_va=np.expand_dims(unscaled_y_test1_va, -1)
unscaled_y_test2_va = unscaled_y2[n:,4]
unscaled_y_test2_va=np.expand_dims(unscaled_y_test2_va, -1)
unscaled_y_test3_va = unscaled_y3[n:,4]
unscaled_y_test3_va=np.expand_dims(unscaled_y_test3_va, -1)
unscaled_y_test4_va = unscaled_y4[n:,4]
unscaled_y_test4_va=np.expand_dims(unscaled_y_test4_va, -1)

print(ohlcv_train.shape)
print(ohlcv_test.shape)
print(tech_ind_train.shape)
print(tech_ind_test.shape)
print(y_train.shape)
print(y_test.shape)

print(ohlcv_train[0,49])
print(y_train[0])


# model architecture
#################################################################
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

# the first branch operates on the first input
x_hi = LSTM(50)(lstm_input)
#x = Dropout(0.2)(x)
lstm_branch_hi = Model(inputs=lstm_input, outputs=x_hi)

# the second branch opreates on the second input
y_hi = Dense(20, activation="relu")(dense_input)
#y = Dropout(0.2)(y)
technical_indicators_branch_hi = Model(inputs=dense_input, outputs=y_hi)

# the first branch operates on the first input
x_lo = LSTM(50)(lstm_input)
#x = Dropout(0.2)(x)
lstm_branch_lo = Model(inputs=lstm_input, outputs=x_lo)

# the second branch opreates on the second input
y_lo = Dense(20, activation="relu")(dense_input)
#y = Dropout(0.2)(y)
technical_indicators_branch_lo = Model(inputs=dense_input, outputs=y_lo)

# the first branch operates on the first input
x_cl = LSTM(50)(lstm_input)
#x = Dropout(0.2)(x)
lstm_branch_cl = Model(inputs=lstm_input, outputs=x_cl)

# the second branch opreates on the second input
y_cl = Dense(20, activation="relu")(dense_input)
#y = Dropout(0.2)(y)
technical_indicators_branch_cl = Model(inputs=dense_input, outputs=y_cl)

# the first branch operates on the first input
x_va = LSTM(50)(lstm_input)
#x = Dropout(0.2)(x)
lstm_branch_va = Model(inputs=lstm_input, outputs=x_va)

# the second branch opreates on the second input
y_va = Dense(20, activation="relu")(dense_input)
#y = Dropout(0.2)(y)
technical_indicators_branch_va = Model(inputs=dense_input, outputs=y_va)

#################################################################

# combine the output of the two branches
combined = concatenate([lstm_branch.output, technical_indicators_branch.output])
z0 = Dense(64, activation="sigmoid")(combined)
z = Dense(1, activation="linear")(z0)
# our model will accept the inputs of the two branches and
# then output a single value
model = Model(inputs=[lstm_branch.input, technical_indicators_branch.input], outputs=z)
#model.summary()

combined_hi = concatenate([lstm_branch_hi.output, technical_indicators_branch_hi.output])
z0_hi = Dense(64, activation="sigmoid")(combined_hi)
z_hi = Dense(1, activation="linear")(z0_hi)
# our model will accept the inputs of the two branches and
# then output a single value
model_hi = Model(inputs=[lstm_branch_hi.input, technical_indicators_branch_hi.input], outputs=z_hi)
#model_hi.summary()

combined_lo = concatenate([lstm_branch_lo.output, technical_indicators_branch_lo.output])
z0_lo = Dense(64, activation="sigmoid")(combined_lo)
z_lo = Dense(1, activation="linear")(z0_lo)
# our model will accept the inputs of the two branches and
# then output a single value
model_lo = Model(inputs=[lstm_branch_lo.input, technical_indicators_branch_lo.input], outputs=z_lo)
#model_lo.summary()

combined_cl = concatenate([lstm_branch_cl.output, technical_indicators_branch_cl.output])
z0_cl = Dense(64, activation="sigmoid")(combined_cl)
z_cl = Dense(1, activation="linear")(z0_cl)
# our model will accept the inputs of the two branches and
# then output a single value
model_cl = Model(inputs=[lstm_branch_cl.input, technical_indicators_branch_cl.input], outputs=z_cl)
#model_cl.summary()

# combine the output of the two branches
combined_va = concatenate([lstm_branch_va.output, technical_indicators_branch_va.output])
z0_va = Dense(64, activation="sigmoid")(combined_va)
z_va = Dense(1, activation="linear")(z0_va)
# our model will accept the inputs of the two branches and
# then output a single value
model_va = Model(inputs=[lstm_branch_va.input, technical_indicators_branch_va.input], outputs=z_va)
#model_va.summary()

#################################################################
combined0_lv1 = concatenate([model.output,model_hi.output,model_lo.output,model_cl.output,model_va.output], axis=1)
combined_lv1 = Reshape((1,5))(combined0_lv1)
print(combined_lv1.shape.as_list())
#print(combined_lv01.shape.as_list())
combined_lvl1 = concatenate([lstm_branch.input,combined_lv1], axis=1)
print(combined_lvl1.shape.as_list())

#x_lvl1_input = Input(shape=(51, 5), name='x_lvl1_input')
# the first branch operates on the first input
x0_lvl1 = LSTM(51, return_sequences=True)(combined_lvl1)
x_lvl1 = LSTM(51)(x0_lvl1)
#x = Dropout(0.2)(x)
#lstm_branch_lvl1 = Model(inputs=[lstm_branch.input, technical_indicators_branch.input], outputs=x_lvl1)

# the second branch opreates on the second input
y0_lvl1 = Dense(20, activation="relu")(dense_input)
y_lvl1 = Dense(20, activation="relu")(y0_lvl1)
#y = Dropout(0.2)(y)
#technical_indicators_branch_lvl1 = Model(inputs=dense_input, outputs=y_lvl1)

combined_lvl1_N = concatenate([x_lvl1, y_lvl1])
z0_lvl1 = Dense(64, activation="sigmoid")(combined_lvl1_N)
z_lvl1 = Dense(1, activation="linear")(z0_lvl1)

model_lvl1 = Model(inputs=[lstm_input, dense_input], outputs=z_lvl1)
#################################################################

modelN = Model(inputs=[lstm_input, dense_input], outputs=[model.output,model_hi.output,model_lo.output,model_cl.output,model_va.output,model_lvl1.output])
# modelN = Model(inputs=[lstm_input, dense_input], outputs=[model.output,model_hi.output,model_lo.output,model_cl.output,model_va.output])
# modelN = Model(inputs=[lstm_input, dense_input], outputs=[model_lvl1.output])
modelN.summary()

# adam = optimizers.Adam(lr=0.0005)
adam = optimizers.Adam()
modelN.compile(loss=['mae','mae','mae','mae','mae','mae'], optimizer=adam)
# modelN.compile(loss=['mae','mae','mae','mae','mae'], optimizer=adam)

#history = modelN.fit([ohlcv_train, tech_ind_train], [y_train,y_train_hi,y_train_lo,y_train_cl,y_train_va, y_train1], batch_size=1024, epochs=50, shuffle=True, validation_split=0.1)
history = modelN.fit([ohlcv_train, tech_ind_train], [y_train1,y_train1_hi,y_train1_lo,y_train1_cl,y_train1_va, y_train2], epochs=50, shuffle=True, validation_split=0.1)
# history = modelN.fit([ohlcv_train, tech_ind_train], [y_train,y_train_hi,y_train_lo,y_train_cl,y_train_va], epochs=50, shuffle=True, validation_split=0.1)
# history = modelN.fit([ohlcv_train, tech_ind_train], [y_train1], epochs=50, shuffle=True, validation_split=0.1)
# history = modelN.fit([ohlcv_train, tech_ind_train], [y_train], epochs=50, shuffle=True, validation_split=0.1)

y_predicted,y_predicted_hi,y_predicted_lo,y_predicted_cl,y_predicted_va,y_predicted1 = modelN.predict([ohlcv_histories, technical_indicators])
# y_predicted,y_predicted_hi,y_predicted_lo,y_predicted_cl,y_predicted_va = modelN.predict([ohlcv_histories, technical_indicators])
# y_predicted1 = modelN.predict([ohlcv_histories, technical_indicators])

#################################################################

y_predicted = y_normaliser.inverse_transform(y_predicted)
y_predicted1 = y_normaliser.inverse_transform(y_predicted1)
y_train = y_normaliser.inverse_transform(y_train)
y_train1 = y_normaliser.inverse_transform(y_train1)
import matplotlib.pyplot as plt
plt.gcf().set_size_inches(22, 15, forward=True)
start = 0
end = -1
real = plt.plot(np.expand_dims(y_train[start:end,0],-1), label='real')
real1 = plt.plot(np.expand_dims(y_train1[start:end,0],-1), label='real1')
pred = plt.plot(y_predicted[start:end], label='predicted')
pred1 = plt.plot(y_predicted1[start:end], label='predicted1')
plt.legend(['Real', 'Real1', 'Predicted', 'Predicted1'])
plt.show()

y_predicted_hi = y_normaliser.inverse_transform(y_predicted_hi)
plt.gcf().set_size_inches(22, 15, forward=True)
real = plt.plot(np.expand_dims(unscaled_y[start:end,1],-1), label='real_hi')
pred = plt.plot(y_predicted_hi[start:end], label='predicted_hi')
plt.legend(['Real_hi', 'Predicted_hi'])
plt.show()

y_predicted_lo = y_normaliser.inverse_transform(y_predicted_lo)
plt.gcf().set_size_inches(22, 15, forward=True)
real = plt.plot(np.expand_dims(unscaled_y[start:end,2],-1), label='real_lo')
pred = plt.plot(y_predicted_lo[start:end], label='predicted_lo')
plt.legend(['Real_lo', 'Predicted_lo'])
plt.show()

y_predicted_cl = y_normaliser.inverse_transform(y_predicted_cl)
plt.gcf().set_size_inches(22, 15, forward=True)
real = plt.plot(np.expand_dims(unscaled_y[start:end,3],-1), label='real_cl')
pred = plt.plot(y_predicted_cl[start:end], label='predicted_cl')
plt.legend(['Real_cl', 'Predicted_cl'])
plt.show()

y_predicted_va = y_normaliser_volume.inverse_transform(y_predicted_va)
plt.gcf().set_size_inches(22, 15, forward=True)
real = plt.plot(np.expand_dims(unscaled_y[start:end,4],-1), label='real_va')
pred = plt.plot(y_predicted_va[start:end], label='predicted_va')
plt.legend(['Real_va', 'Predicted_va'])
plt.show()

plt.gcf().set_size_inches(22, 15, forward=True)
real = plt.plot(np.expand_dims(unscaled_y[start:end,0],-1), label='real')
pred = plt.plot(y_predicted[start:end], label='predicted')
pred = plt.plot(y_predicted_hi[start:end], label='predicted_hi')
pred = plt.plot(y_predicted_lo[start:end], label='predicted_lo')
pred = plt.plot(y_predicted_cl[start:end], label='predicted_cl')
plt.legend(['Real', 'Predicted', 'Predicted_hi', 'Predicted_lo', 'Predicted_cl'])
plt.show()

sys.exit()

# combined1 = concatenate([lstm_branch.output, model.output,technical_indicators_branch.output])
# z1 = Dense(64, activation="sigmoid")(combined1)
# z1 = Dense(1, activation="linear")(z1)
# # our model will accept the inputs of the two branches and
# # then output a single value
# model1 = Model(inputs=[lstm_branch.input, technical_indicators_branch.input], outputs=z1)
# model1.summary()

# combined2 = concatenate([lstm_branch.output, model.output, model1.output,technical_indicators_branch.output])
# z2 = Dense(64, activation="sigmoid")(combined2)
# z2 = Dense(1, activation="linear")(z2)
# # our model will accept the inputs of the two branches and
# # then output a single value
# model2 = Model(inputs=[lstm_branch.input, technical_indicators_branch.input], outputs=z2)
# model2.summary()

# combined3 = concatenate([lstm_branch.output, model.output, model1.output, model2.output, technical_indicators_branch.output])
# z3 = Dense(64, activation="sigmoid")(combined3)
# z3 = Dense(1, activation="linear")(z3)
# # our model will accept the inputs of the two branches and
# # then output a single value
# model3 = Model(inputs=[lstm_branch.input, technical_indicators_branch.input], outputs=z3)
# model3.summary()

# modelN = Model(inputs=[lstm_branch.input, technical_indicators_branch.input], outputs=[model.output,model1.output,model2.output,model3.output])
# modelN.summary()

# # adam = optimizers.Adam(lr=0.0005)
# adam = optimizers.Adam()
# modelN.compile(loss=['mae', 'mae', 'mae', 'mae'], optimizer=adam)

# history = modelN.fit([ohlcv_train, tech_ind_train], [y_train, y_train1, y_train2, y_train3], batch_size=1024, epochs=50, shuffle=True, validation_split=0.1)

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

unscaled_y = unscaled_y[:,0]
unscaled_y=np.expand_dims(unscaled_y, -1)
unscaled_y1 = unscaled_y1[:,0]
unscaled_y1=np.expand_dims(unscaled_y1, -1)
unscaled_y2 = unscaled_y2[:,0]
unscaled_y2=np.expand_dims(unscaled_y2, -1)
unscaled_y3 = unscaled_y3[:,0]
unscaled_y3=np.expand_dims(unscaled_y3, -1)
unscaled_y4 = unscaled_y4[:,0]
unscaled_y4=np.expand_dims(unscaled_y4, -1)


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
plt.legend(['Real', 'real1', 'real2', 'real3'])
plt.show()

from datetime import datetime
model.save(f'technical_model_N.h5')
