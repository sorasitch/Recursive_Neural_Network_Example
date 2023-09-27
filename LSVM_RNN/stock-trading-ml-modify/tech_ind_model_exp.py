import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from keras import optimizers
import numpy as np
np.random.seed(4)
#from tensorflow import set_random_seed
#set_random_seed(4)
from util_exp import csv_to_dataset, history_points
import sys
import os.path


# dataset

# ohlcv_histories, technical_indicators, next_day_open_values, unscaled_y, y_normaliser, _, _, _, _, _, _, _, _, _ = csv_to_dataset('MSFT_daily.csv')
ohlcv_histories_normalised, technical_indicators_normalised, next_day_open_values_normalised, next_day_open_values, y_normaliser_open, next_day_high_values_normalised, next_day_high_values, y_normaliser_high, next_day_low_values_normalised, next_day_low_values, y_normaliser_low, next_day_close_values_normalised, next_day_close_values, y_normaliser_close, next_day_volume_values_normalised, next_day_volume_values, y_normaliser_volume= csv_to_dataset('MSFT_daily.csv')
# sys.exit()

test_split = 0.9
n = int(ohlcv_histories_normalised.shape[0] * test_split)

ohlcv_train = ohlcv_histories_normalised[:n]
tech_ind_train = technical_indicators_normalised[:n]
y_train_open = next_day_open_values_normalised[:n]
y_train_high = next_day_high_values_normalised[:n]
y_train_low = next_day_low_values_normalised[:n]
y_train_close = next_day_close_values_normalised[:n]
y_train_volume = next_day_volume_values_normalised[:n]

ohlcv_test = ohlcv_histories_normalised[n:]
tech_ind_test = technical_indicators_normalised[n:]
y_test_open = next_day_open_values_normalised[n:]
y_test_high = next_day_high_values_normalised[n:]
y_test_low = next_day_low_values_normalised[n:]
y_test_close = next_day_close_values_normalised[n:]
y_test_volume = next_day_volume_values_normalised[n:]

unscaled_y_test_open = next_day_open_values_normalised[n:]
unscaled_y_test_high = next_day_high_values_normalised[n:]
unscaled_y_test_low = next_day_low_values_normalised[n:]
unscaled_y_test_close = next_day_close_values_normalised[n:]
unscaled_y_test_volume = next_day_volume_values_normalised[n:]


print(technical_indicators_normalised.shape)
print(ohlcv_histories_normalised.shape)
# print(ohlcv_train.shape)
# print(ohlcv_test.shape)
# print(tech_ind_train.shape)
# print(tech_ind_test.shape)
# print(y_train_open.shape)
# print(y_test_open.shape)
# print(ohlcv_train[:][0])


# model architecture
if os.path.exists(f'technical_model_volume.h5') and os.path.isfile(f'technical_model_volume.h5') :
    pass
    # load
    model_open=keras.models.load_model(f'technical_model_open.h5',compile=False)
    model_high=keras.models.load_model(f'technical_model_high.h5',compile=False)
    model_low=keras.models.load_model(f'technical_model_low.h5',compile=False)
    model_close=keras.models.load_model(f'technical_model_close.h5',compile=False)
    model_volume=keras.models.load_model(f'technical_model_volume.h5',compile=False)
else :
    pass
    # define two sets of inputs
    lstm_input = Input(shape=(history_points, 5)) #, name='lstm_input')
    dense_input = Input(shape=(technical_indicators_normalised.shape[1],)) #, name='tech_input')

    # the first branch operates on the first input
    x_open = LSTM(50)(lstm_input)
    # x_open = Dropout(0.2)(x_open)
    x_open = Dense(128, activation="relu")(x_open)
    x_open = Dense(64, activation="relu")(x_open)

    x_high = LSTM(50)(lstm_input)
    # x_high = Dropout(0.2)(x_high)
    x_high = Dense(128, activation="relu")(x_high)
    x_high = Dense(64, activation="relu")(x_high)

    x_low = LSTM(50)(lstm_input)
    # x_low = Dropout(0.2)(x_low)
    x_low = Dense(128, activation="relu")(x_low)
    x_low = Dense(64, activation="relu")(x_low)

    x_close = LSTM(50)(lstm_input)
    # x_close = Dropout(0.2)(x_close)
    x_close = Dense(128, activation="relu")(x_close)
    x_close = Dense(64, activation="relu")(x_close)

    x_volume = LSTM(50)(lstm_input)
    # x_volume = Dropout(0.2)(x_volume)
    x_volume = Dense(128, activation="relu")(x_volume)
    x_volume = Dense(64, activation="relu")(x_volume)

    lstm_branch_open = Model(inputs=lstm_input, outputs=x_open)
    lstm_branch_high = Model(inputs=lstm_input, outputs=x_high)
    lstm_branch_low = Model(inputs=lstm_input, outputs=x_low)
    lstm_branch_close = Model(inputs=lstm_input, outputs=x_close)
    lstm_branch_volume = Model(inputs=lstm_input, outputs=x_volume)

    # the second branch opreates on the second input
    y_open = Dense(128, activation="relu")(dense_input)
    # y_open = Dropout(0.2)(y_open)
    y_open = Dense(64, activation="relu")(y_open)

    y_high = Dense(128, activation="relu")(dense_input)
    # y_high = Dropout(0.2)(y_high)
    y_high = Dense(64, activation="relu")(y_high)

    y_low = Dense(128, activation="relu")(dense_input)
    # y_low = Dropout(0.2)(y_low)
    y_low = Dense(64, activation="relu")(y_low)

    y_close = Dense(128, activation="relu")(dense_input)
    # y_close = Dropout(0.2)(y_close)
    y_close = Dense(64, activation="relu")(y_close)

    y_volume = Dense(128, activation="relu")(dense_input)
    # y_volume = Dropout(0.2)(y_volume)
    y_volume = Dense(64, activation="relu")(y_volume)

    technical_indicators_branch_open = Model(inputs=dense_input, outputs=y_open)
    technical_indicators_branch_high = Model(inputs=dense_input, outputs=y_high)
    technical_indicators_branch_low = Model(inputs=dense_input, outputs=y_low)
    technical_indicators_branch_close = Model(inputs=dense_input, outputs=y_close)
    technical_indicators_branch_volume = Model(inputs=dense_input, outputs=y_volume)

    # combine the output of the two branches
    combined_open = concatenate([lstm_branch_open.output, technical_indicators_branch_open.output])
    combined_high = concatenate([lstm_branch_high.output, technical_indicators_branch_high.output])
    combined_low = concatenate([lstm_branch_low.output, technical_indicators_branch_low.output])
    combined_close = concatenate([lstm_branch_close.output, technical_indicators_branch_close.output])
    combined_volume = concatenate([lstm_branch_volume.output, technical_indicators_branch_volume.output])

    z_open = Dense(128, activation="relu")(combined_open)
    z_open = Dense(64, activation="relu")(z_open)
    z_open = Dense(1, activation="linear")(z_open)

    z_high = Dense(128, activation="relu")(combined_high)
    z_high = Dense(64, activation="relu")(z_high)
    z_high = Dense(1, activation="linear")(z_high)

    z_low = Dense(128, activation="relu")(combined_low)
    z_low = Dense(64, activation="relu")(z_low)
    z_low = Dense(1, activation="linear")(z_low)

    z_close = Dense(128, activation="relu")(combined_close)
    z_close = Dense(64, activation="relu")(z_close)
    z_close = Dense(1, activation="linear")(z_close)

    z_volume = Dense(128, activation="relu")(combined_volume)
    z_volume = Dense(64, activation="relu")(z_volume)
    z_volume = Dense(1, activation="linear")(z_volume)


    # our model will accept the inputs of the two branches and
    # then output a single value
    # 

    # model_high = Model(inputs=[lstm_branch_high.input, technical_indicators_branch_high.input], outputs=z_high)
    # adam_high = optimizers.Adam(lr=0.0005)
    # model_high.compile(optimizer=adam_high, loss='mse')
    # # model_high.fit(x=[ohlcv_train, tech_ind_train], y=y_train_high, batch_size=32, epochs=50, shuffle=True, validation_split=0.1)    
    # model_high.fit(x=[ohlcv_train, tech_ind_train], y=y_train_high, batch_size=1024, epochs=50, validation_split=0.1)    

    # model_open = Model(inputs=[lstm_branch_open.input, technical_indicators_branch_open.input], outputs=z_open)
    # adam_open = optimizers.Adam(lr=0.0005)
    # # model_open.set_weights(model_high.get_weights())
    # model_open.compile(optimizer=adam_open, loss='mse')
    # # model_open.fit(x=[ohlcv_train, tech_ind_train], y=y_train_open, batch_size=32, epochs=50, shuffle=True, validation_split=0.1)
    # model_open.fit(x=[ohlcv_train, tech_ind_train], y=y_train_open, batch_size=1024, epochs=50, validation_split=0.1) 

    # model_low = Model(inputs=[lstm_branch_low.input, technical_indicators_branch_low.input], outputs=z_low)
    # adam_low = optimizers.Adam(lr=0.0005)
    # # model_low.set_weights(model_high.get_weights())
    # model_low.compile(optimizer=adam_low, loss='mse')
    # # model_low.fit(x=[ohlcv_train, tech_ind_train], y=y_train_low, batch_size=32, epochs=50, shuffle=True, validation_split=0.1)
    # model_low.fit(x=[ohlcv_train, tech_ind_train], y=y_train_low, batch_size=1024, epochs=50, validation_split=0.1)       

    # model_close = Model(inputs=[lstm_branch_close.input, technical_indicators_branch_close.input], outputs=z_close)
    # adam_close = optimizers.Adam(lr=0.0005)
    # # model_close.set_weights(model_high.get_weights())
    # model_close.compile(optimizer=adam_close, loss='mse')
    # # model_close.fit(x=[ohlcv_train, tech_ind_train], y=y_train_close, batch_size=32, epochs=50, shuffle=True, validation_split=0.1)
    # model_close.fit(x=[ohlcv_train, tech_ind_train], y=y_train_close, batch_size=1024, epochs=50, validation_split=0.1)       

    # model_volume = Model(inputs=[lstm_branch_volume.input, technical_indicators_branch_volume.input], outputs=z_volume)
    # adam_volume = optimizers.Adam(lr=0.0005)
    # model_volume.compile(optimizer=adam_volume, loss='mse')
    # # model_volume.fit(x=[ohlcv_train, tech_ind_train], y=y_train_volume, batch_size=32, epochs=50, shuffle=True, validation_split=0.1)    
    # model_volume.fit(x=[ohlcv_train, tech_ind_train], y=y_train_volume, batch_size=1024, epochs=50, validation_split=0.1)        


    model_high = Model(inputs=[lstm_branch_high.input, technical_indicators_branch_high.input], outputs=z_high)
    adam_high = optimizers.Adam(lr=0.0005)
    model_high.compile(optimizer=adam_high, loss='mse')
    # model_high.fit(x=[ohlcv_train, tech_ind_train], y=y_train_high, batch_size=32, epochs=50, shuffle=True, validation_split=0.1)    
    model_high.fit(x=[ohlcv_train, tech_ind_train], y=y_train_high, batch_size=1024, epochs=50, validation_split=0.1)    

    model_low = Model(inputs=[lstm_branch_low.input, technical_indicators_branch_low.input], outputs=z_low)
    adam_low = optimizers.Adam(lr=0.0005)
    # model_low.set_weights(model_high.get_weights())
    model_low.compile(optimizer=adam_low, loss='mse')
    # model_low.fit(x=[ohlcv_train, tech_ind_train], y=y_train_low, batch_size=32, epochs=50, shuffle=True, validation_split=0.1)
    model_low.fit(x=[ohlcv_train, tech_ind_train], y=y_train_low, batch_size=1024, epochs=50, validation_split=0.1)    

    model_open = Model(inputs=[lstm_branch_open.input, technical_indicators_branch_open.input], outputs=z_open)
    adam_open = optimizers.Adam(lr=0.0005)
    # model_open.set_weights(model_high.get_weights())
    model_open.compile(optimizer=adam_open, loss='mse')
    # model_open.fit(x=[ohlcv_train, tech_ind_train], y=y_train_open, batch_size=32, epochs=50, shuffle=True, validation_split=0.1)
    model_open.fit(x=[ohlcv_train, tech_ind_train], y=y_train_open, batch_size=1024, epochs=50, validation_split=0.1)    

    model_close = Model(inputs=[lstm_branch_close.input, technical_indicators_branch_close.input], outputs=z_close)
    adam_close = optimizers.Adam(lr=0.0005)
    # model_close.set_weights(model_high.get_weights())
    model_close.compile(optimizer=adam_close, loss='mse')
    # model_close.fit(x=[ohlcv_train, tech_ind_train], y=y_train_close, batch_size=32, epochs=50, shuffle=True, validation_split=0.1)
    model_close.fit(x=[ohlcv_train, tech_ind_train], y=y_train_close, batch_size=1024, epochs=50, validation_split=0.1)       

    model_volume = Model(inputs=[lstm_branch_volume.input, technical_indicators_branch_volume.input], outputs=z_volume)
    adam_volume = optimizers.Adam(lr=0.0005)
    model_volume.compile(optimizer=adam_volume, loss='mse')
    # model_volume.fit(x=[ohlcv_train, tech_ind_train], y=y_train_volume, batch_size=32, epochs=50, shuffle=True, validation_split=0.1)    
    model_volume.fit(x=[ohlcv_train, tech_ind_train], y=y_train_volume, batch_size=1024, epochs=50, validation_split=0.1)        


    from datetime import datetime
    model_open.save(f'technical_model_open.h5')
    model_high.save(f'technical_model_high.h5')
    model_low.save(f'technical_model_low.h5')
    model_close.save(f'technical_model_close.h5')
    model_volume.save(f'technical_model_volume.h5')


# evaluation
y_test_predicted_open = model_open.predict([ohlcv_test, tech_ind_test])
y_test_predicted_open = y_normaliser_open.inverse_transform(y_test_predicted_open)
y_predicted_open = model_open.predict([ohlcv_histories_normalised, technical_indicators_normalised])
y_predicted_open = y_normaliser_open.inverse_transform(y_predicted_open)

y_test_predicted_high = model_high.predict([ohlcv_test, tech_ind_test])
y_test_predicted_high = y_normaliser_high.inverse_transform(y_test_predicted_high)
y_predicted_high = model_high.predict([ohlcv_histories_normalised, technical_indicators_normalised])
y_predicted_high = y_normaliser_high.inverse_transform(y_predicted_high)

y_test_predicted_low = model_low.predict([ohlcv_test, tech_ind_test])
y_test_predicted_low = y_normaliser_low.inverse_transform(y_test_predicted_low)
y_predicted_low = model_low.predict([ohlcv_histories_normalised, technical_indicators_normalised])
y_predicted_low = y_normaliser_low.inverse_transform(y_predicted_low)

y_test_predicted_close = model_close.predict([ohlcv_test, tech_ind_test])
y_test_predicted_close = y_normaliser_close.inverse_transform(y_test_predicted_close)
y_predicted_close = model_close.predict([ohlcv_histories_normalised, technical_indicators_normalised])
y_predicted_close = y_normaliser_close.inverse_transform(y_predicted_close)

y_test_predicted_volume = model_volume.predict([ohlcv_test, tech_ind_test])
y_test_predicted_volume = y_normaliser_volume.inverse_transform(y_test_predicted_volume)
y_predicted_volume = model_volume.predict([ohlcv_histories_normalised, technical_indicators_normalised])
y_predicted_volume = y_normaliser_volume.inverse_transform(y_predicted_volume)

# y_test_predicted = []
# for open,high,low,close,volume in zip(y_test_predicted_open,y_test_predicted_high,y_test_predicted_low,y_test_predicted_close,y_test_predicted_volume):
#     pass
#     # y_test_predicted.append(np.array([open,high,low,close,volume]))
#     # open.copy()
#     y_test_predicted.append([open[-1],high[-1],low[-1],close[-1],volume[-1]])

# y_test_predicted = np.array(y_test_predicted)
# print(y_test_predicted[:][:10])
# y_test_predicted = y_normaliser_high.inverse_transform(y_test_predicted)
# print(y_test_predicted[:][:10])
# sys.exit()

# print(y_test_predicted_open.shape)
# print(y_test_predicted_open[:][:10])
# print(y_test_predicted_high.shape)
# print(y_test_predicted_high[:][:10])

next_day_values_normalised = np.append(next_day_open_values_normalised,next_day_high_values_normalised,axis=1)
next_day_values_normalised = np.append(next_day_values_normalised,next_day_low_values_normalised,axis=1)
next_day_values_normalised = np.append(next_day_values_normalised,next_day_close_values_normalised,axis=1)
next_day_values_normalised = np.append(next_day_values_normalised,next_day_volume_values_normalised,axis=1)


data_normalised = np.append(y_predicted_open,y_predicted_high,axis=1)
data_normalised = np.append(data_normalised,y_predicted_low,axis=1)
data_normalised = np.append(data_normalised,y_predicted_close,axis=1)
data_normalised = np.append(data_normalised,y_predicted_volume,axis=1)

print(next_day_values_normalised.shape)
print(next_day_values_normalised[:][:10])
print(data_normalised.shape)
print(data_normalised[:][:10])

# sys.exit()


assert unscaled_y_test_open.shape == y_test_predicted_open.shape
real_mse = np.mean(np.square(unscaled_y_test_open - y_test_predicted_open))
scaled_mse = real_mse / (np.max(unscaled_y_test_open) - np.min(unscaled_y_test_open)) * 100
print(real_mse)
print(scaled_mse)

import matplotlib.pyplot as plt

plt.gcf().set_size_inches(22, 15, forward=True)

start = 0
end = -1

# real = plt.plot(unscaled_y_test_open[start:end], label='real')
# pred_open = plt.plot(y_test_predicted_open[start:end], label='predicted_open')
# pred_high = plt.plot(y_test_predicted_high[start:end], label='predicted_high')
# pred_low = plt.plot(y_test_predicted_low[start:end], label='predicted_low')
# pred_close = plt.plot(y_test_predicted_close[start:end], label='predicted_close')

real = plt.plot(next_day_open_values[start:end], label='real')
pred_open = plt.plot(y_predicted_open[start:end], label='predicted_open')
pred_high = plt.plot(y_predicted_high[start:end], label='predicted_high')
pred_low = plt.plot(y_predicted_low[start:end], label='predicted_low')
pred_close = plt.plot(y_predicted_close[start:end], label='predicted_close')


plt.legend(['Real', 'Predicted_open', 'Predicted_high', 'Predicted_low', 'Predicted_close'])
plt.show()

plt.gcf().set_size_inches(22, 15, forward=True)

# real_volume = plt.plot(unscaled_y_test_volume[start:end], label='real_volume')
# pred_volume = plt.plot(y_test_predicted_volume[start:end], label='predicted_volume')

real_volume = plt.plot(next_day_volume_values[start:end], label='real_volume')
pred_volume = plt.plot(y_predicted_volume[start:end], label='predicted_volume')

plt.legend(['Real_volume', 'Predicted_volume'])
plt.show()



def calc_ema(values, time_period):
    # https://www.investopedia.com/ask/answers/122314/what-exponential-moving-average-ema-formula-and-how-ema-calculated.asp
    sma = np.mean(values[:, 3])
    ema_values = [sma]
    k = 2 / (1 + time_period)
    for i in range(len(values) - time_period, len(values)):
        close = values[i][3]
        ema_values.append(close * k + ema_values[-1] * (1 - k))
    return ema_values[-1]

m = 4150
l = 100
x_his_0 = ohlcv_histories_normalised[m]
x_his = x_his_0.reshape((1, x_his_0.shape[0], x_his_0.shape[1]))
x_ind_0 = technical_indicators_normalised[m]
x_ind = x_ind_0.reshape((1, x_ind_0.shape[0]))

for i in range(l) :
    pass
    # print(x_his)
    # print(x_ind)
    # print(x_his.shape)
    # print(x_ind.shape)

    y_open = model_open.predict([x_his, x_ind])
    y_open_1 = y_normaliser_open.inverse_transform(y_open)
    y_high = model_high.predict([x_his, x_ind])
    y_high_1 = y_normaliser_high.inverse_transform(y_high)
    y_low = model_low.predict([x_his, x_ind])
    y_low_1 = y_normaliser_low.inverse_transform(y_low)
    y_close = model_close.predict([x_his, x_ind])
    y_close_1 = y_normaliser_close.inverse_transform(y_close)
    y_volume = model_volume.predict([x_his, x_ind])
    y_volume_1 = y_normaliser_volume.inverse_transform(y_volume)

    y_normalised = np.append(y_open,y_high,axis=1)
    y_normalised = np.append(y_normalised,y_low,axis=1)
    y_normalised = np.append(y_normalised,y_close,axis=1)
    y_normalised = np.append(y_normalised,y_volume,axis=1)
    # print(y_normalised)

    x_his_0 = np.append(x_his_0, y_normalised, axis=0)
    # print(x_his_0.shape)
    x_his_1 = x_his_0[-50:]
    # print(x_his_1[-1])

    sma = np.mean(x_his_1[:, 3])
    macd = calc_ema(x_his_1, 12) - calc_ema(x_his_1, 26)
    x_ind_1=np.array([sma,macd])
    # print(x_ind_1.shape)
    # print(x_ind_1)
    x_his = x_his_1.reshape((1, x_his_1.shape[0], x_his_1.shape[1]))
    x_ind = x_ind_1.reshape((1, x_ind_1.shape[0]))

    # exp use real one
    x_his_3 = ohlcv_histories_normalised[i+1+m]
    x_his = x_his_3.reshape((1, x_his_3.shape[0], x_his_3.shape[1]))
    x_ind_3 = technical_indicators_normalised[i+1+m] 
    x_ind = x_ind_3.reshape((1, x_ind_3.shape[0]))

print(x_his_0.shape)
# x_his = x_his_0.reshape((1, x_his_0.shape[0], x_his_0.shape[1]))
op = np.array(x_his_0[:, 0])
op = np.expand_dims(op, -1)
hi = np.array(x_his_0[:, 1])
hi = np.expand_dims(hi, -1)
op = y_normaliser_open.inverse_transform(op)
hi = y_normaliser_open.inverse_transform(hi)

plt.gcf().set_size_inches(22, 15, forward=True)
plt.plot(next_day_open_values[m-50:m-50+50+l+200], label='real')
plt.plot(op, label='open')
plt.plot(hi, label='high')
plt.legend(['real', 'open', 'high'])
plt.show()

