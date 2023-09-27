import pandas as pd
from sklearn import preprocessing
import numpy as np

history_points = 50
N=1
End = history_points*N


def csv_to_dataset(csv_path):
    data = pd.read_csv(csv_path)
    data = data.drop('date', axis=1)
    data = data.drop(0, axis=0)

    data = data.values

    data_normaliser = preprocessing.MinMaxScaler()
    data_normalised = data_normaliser.fit_transform(data)

    # y_normaliser = preprocessing.MinMaxScaler()
    # y_normaliser.fit(data)
    # data_normalised = y_normaliser.transform(data)

    ohlcv_histories_normalised = np.array([data_normalised[:,0:][i:i+End:N].copy() for i in range(len(data_normalised) - End)])
    print(ohlcv_histories_normalised.shape)
    print(ohlcv_histories_normalised[0,0,:])
    print(ohlcv_histories_normalised[0,-1,:])
    print(ohlcv_histories_normalised[End,0,:])
    print(ohlcv_histories_normalised[End,-1,:])
    next_day_open_values_normalised = np.array([data_normalised[:, 0][i + End].copy() for i in range(len(data_normalised) - End)])
    # next_day_open_values_normalised = np.array([data_normalised[:, 0][i + End - N].copy() for i in range(len(data_normalised) - End)])  # same at the i + End
    print(next_day_open_values_normalised.shape)
    print(next_day_open_values_normalised[0])
    next_day_open_values_normalised = np.expand_dims(next_day_open_values_normalised, -1)

    next_day_open_values = np.array([data[:, 0][i + End].copy() for i in range(len(data) - End)])
    # next_day_open_values = np.array([data[:, 0][i + End - N].copy() for i in range(len(data) - End)])  # same at the i + End
    print(next_day_open_values.shape)
    next_day_open_values = np.expand_dims(next_day_open_values, -1)

    ohlcv_histories_normalised_ind = np.array([data_normalised[i:i + End].copy() for i in range(len(data_normalised) - End)])
    print(ohlcv_histories_normalised_ind.shape)

    # # using the last {history_points} open close high low volume data points, predict the next open value
    # ohlcv_histories_normalised = np.array([data_normalised[i:i + history_points].copy() for i in range(len(data_normalised) - history_points - N)])
    # # print(ohlcv_histories_normalised.shape)
    # next_day_open_values_normalised = np.array([data_normalised[:, 0][i + history_points + N].copy() for i in range(len(data_normalised) - history_points - N)])
    # next_day_open_values_normalised = np.expand_dims(next_day_open_values_normalised, -1)

    # next_day_open_values = np.array([data[:, 0][i + history_points + N].copy() for i in range(len(data) - history_points - N)])
    # next_day_open_values = np.expand_dims(next_day_open_values, -1) 


    # y_normaliser = preprocessing.MinMaxScaler()
    # y_normaliser.fit(next_day_open_values)

    y_normaliser = preprocessing.MinMaxScaler()
    data_op = np.expand_dims(data[:, 0], -1)
    y_normaliser.fit(data_op)

    def calc_ema(values, time_period):
        # https://www.investopedia.com/ask/answers/122314/what-exponential-moving-average-ema-formula-and-how-ema-calculated.asp
        sma = np.mean(values[:, 3])
        ema_values = [sma]
        k = 2 / (1 + time_period)
        for i in range(len(his) - time_period, len(his)):
            close = his[i][3]
            ema_values.append(close * k + ema_values[-1] * (1 - k))
        return ema_values[-1]

    technical_indicators = []
    # for his in ohlcv_histories_normalised:
    #     # note since we are using his[3] we are taking the SMA of the closing price
    #     sma = np.mean(his[:, 3])
    #     macd = calc_ema(his, 12) - calc_ema(his, 26)
    #     #technical_indicators.append(np.array([sma]))
    #     technical_indicators.append(np.array([sma,macd,]))
    for his in ohlcv_histories_normalised_ind:
        # note since we are using his[3] we are taking the SMA of the closing price
        sma = np.mean(his[:, 3])
        macd = calc_ema(his, 12) - calc_ema(his, 26)
        #technical_indicators.append(np.array([sma]))
        technical_indicators.append(np.array([sma,macd,]))

    technical_indicators = np.array(technical_indicators)
    print(technical_indicators.shape)

    tech_ind_scaler = preprocessing.MinMaxScaler()
    technical_indicators_normalised = tech_ind_scaler.fit_transform(technical_indicators)

    assert ohlcv_histories_normalised.shape[0] == next_day_open_values_normalised.shape[0] == technical_indicators_normalised.shape[0]
    return ohlcv_histories_normalised, technical_indicators_normalised, next_day_open_values_normalised, next_day_open_values, y_normaliser


def multiple_csv_to_dataset(test_set_name):
    import os
    ohlcv_histories = 0
    technical_indicators = 0
    next_day_open_values = 0
    for csv_file_path in list(filter(lambda x: x.endswith('daily.csv'), os.listdir('./'))):
        if not csv_file_path == test_set_name:
            print(csv_file_path)
            if type(ohlcv_histories) == int:
                ohlcv_histories, technical_indicators, next_day_open_values, _, _ = csv_to_dataset(csv_file_path)
            else:
                a, b, c, _, _ = csv_to_dataset(csv_file_path)
                ohlcv_histories = np.concatenate((ohlcv_histories, a), 0)
                technical_indicators = np.concatenate((technical_indicators, b), 0)
                next_day_open_values = np.concatenate((next_day_open_values, c), 0)

    ohlcv_train = ohlcv_histories
    tech_ind_train = technical_indicators
    y_train = next_day_open_values

    ohlcv_test, tech_ind_test, y_test, unscaled_y_test, y_normaliser = csv_to_dataset(test_set_name)

    return ohlcv_train, tech_ind_train, y_train, ohlcv_test, tech_ind_test, y_test, unscaled_y_test, y_normaliser
