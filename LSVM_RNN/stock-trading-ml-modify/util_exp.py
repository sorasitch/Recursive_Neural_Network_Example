import pandas as pd
from sklearn import preprocessing
import numpy as np
import sys

history_points = 50
N=0


def csv_to_dataset(csv_path):
    data = pd.read_csv(csv_path)
    data = data.drop('date', axis=1)
    data = data.drop(0, axis=0)

    data = data.values
    # print(data[:][:10])
    print(data.shape)

    data_pr = np.concatenate((data[:,0], data[:,1], data[:,2],data[:,3]))
    # print(data_pr[:][:10])
    print(data_pr.shape)
    data_pr = np.expand_dims(data_pr, -1)
    data_pr_normaliser = preprocessing.MinMaxScaler()
    # data_pr_normalised = data_normaliser.fit_transform(data_pr)
    # data_pr_normaliser = preprocessing.StandardScaler()
    # data_pr_normaliser.fit(data_pr)
    data_pr_normaliser.fit(data[:,:4])
    data_pr_normalised=data_pr_normaliser.transform(data[:,:4])
    print(data_pr_normalised.shape)
    print(data_pr_normalised[:][:10])

    data_vl = data[:,4]
    # print(data_vl[:][:10])
    print(data_vl.shape)
    data_vl = np.expand_dims(data_vl, -1)
    data_vl_normaliser = preprocessing.MinMaxScaler()
    # data_vl_normalised = data_normaliser.fit_transform(data_vl)
    # data_vl_normaliser = preprocessing.StandardScaler()
    data_vl_normaliser.fit(data_vl)
    data_vl_normalised=data_vl_normaliser.transform(data_vl)
    print(data_vl_normalised.shape)
    print(data_vl_normalised[:][:10])

    data_normalised = np.append(data_pr_normalised,data_vl_normalised,axis=1)
    print(data_normalised.shape)
    print(data_normalised[:][:10])


    # data_normaliser = preprocessing.MinMaxScaler()
    # data_normalised = data_normaliser.fit_transform(data)
    # data_normaliser = preprocessing.StandardScaler()
    # data_normaliser.fit(data)
    # data_normalised=data_normaliser.transform(data)
    # print(data_normalised.shape)
    # print(data_normalised[:][:10])
    # sys.exit()


    # using the last {history_points} open close high low volume data points, predict the next open value
    ohlcv_histories_normalised = np.array([data_normalised[i:i + history_points].copy() for i in range(len(data_normalised) - history_points - N)])
    # print(ohlcv_histories_normalised[:][0])

    next_day_open_values_normalised = np.array([data_normalised[:, 0][i + history_points + N].copy() for i in range(len(data_normalised) - history_points - N)])
    next_day_open_values_normalised = np.expand_dims(next_day_open_values_normalised, -1)

    next_day_open_values = np.array([data[:, 0][i + history_points + N].copy() for i in range(len(data) - history_points - N)])
    next_day_open_values = np.expand_dims(next_day_open_values, -1)

    # y_normaliser_open = preprocessing.MinMaxScaler()
    # y_normaliser_open.fit(next_day_open_values)
    # y_normaliser_open = preprocessing.StandardScaler()
    # y_normaliser_open.fit(next_day_open_values)
    y_normaliser_open = data_pr_normaliser

    next_day_high_values_normalised = np.array([data_normalised[:, 1][i + history_points + N].copy() for i in range(len(data_normalised) - history_points - N)])
    next_day_high_values_normalised = np.expand_dims(next_day_high_values_normalised, -1)

    next_day_high_values = np.array([data[:, 1][i + history_points + N].copy() for i in range(len(data) - history_points - N)])
    next_day_high_values = np.expand_dims(next_day_high_values, -1)

    # y_normaliser_high = preprocessing.MinMaxScaler()
    # y_normaliser_high.fit(next_day_high_values)
    # y_normaliser_high = preprocessing.StandardScaler()
    # y_normaliser_high.fit(next_day_high_values)
    y_normaliser_high = data_pr_normaliser

    next_day_low_values_normalised = np.array([data_normalised[:, 2][i + history_points + N].copy() for i in range(len(data_normalised) - history_points - N)])
    next_day_low_values_normalised = np.expand_dims(next_day_low_values_normalised, -1)

    next_day_low_values = np.array([data[:, 2][i + history_points + N].copy() for i in range(len(data) - history_points - N)])
    next_day_low_values = np.expand_dims(next_day_low_values, -1)

    # y_normaliser_low = preprocessing.MinMaxScaler()
    # y_normaliser_low.fit(next_day_low_values)
    # y_normaliser_low = preprocessing.StandardScaler()
    # y_normaliser_low.fit(next_day_low_values)
    y_normaliser_low = data_pr_normaliser

    next_day_close_values_normalised = np.array([data_normalised[:, 3][i + history_points + N].copy() for i in range(len(data_normalised) - history_points - N)])
    next_day_close_values_normalised = np.expand_dims(next_day_close_values_normalised, -1)

    next_day_close_values = np.array([data[:, 3][i + history_points + N].copy() for i in range(len(data) - history_points - N)])
    next_day_close_values = np.expand_dims(next_day_close_values, -1)

    # y_normaliser_close = preprocessing.MinMaxScaler()
    # y_normaliser_close.fit(next_day_close_values)
    # y_normaliser_close = preprocessing.StandardScaler()
    # y_normaliser_close.fit(next_day_close_values)
    y_normaliser_close = data_pr_normaliser

    next_day_volume_values_normalised = np.array([data_normalised[:, 4][i + history_points + N].copy() for i in range(len(data_normalised) - history_points - N)])
    next_day_volume_values_normalised = np.expand_dims(next_day_volume_values_normalised, -1)

    next_day_volume_values = np.array([data[:, 4][i + history_points + N].copy() for i in range(len(data) - history_points - N)])
    next_day_volume_values = np.expand_dims(next_day_volume_values, -1)

    # y_normaliser_volume = preprocessing.MinMaxScaler()
    # y_normaliser_volume.fit(next_day_volume_values)
    # y_normaliser_volume = preprocessing.StandardScaler()
    # y_normaliser_volume.fit(next_day_volume_values)
    y_normaliser_volume = data_vl_normaliser

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
    for his in ohlcv_histories_normalised:
        # note since we are using his[3] we are taking the SMA of the closing price
        sma = np.mean(his[:, 3])
        macd = calc_ema(his, 12) - calc_ema(his, 26)
        #technical_indicators.append(np.array([sma]))
        technical_indicators.append(np.array([sma,macd,]))

    technical_indicators = np.array(technical_indicators)
    # print(technical_indicators[:][0])

    tech_ind_scaler = preprocessing.MinMaxScaler()
    # tech_ind_scaler = preprocessing.StandardScaler()
    technical_indicators_normalised = tech_ind_scaler.fit_transform(technical_indicators)

    assert ohlcv_histories_normalised.shape[0] == next_day_open_values_normalised.shape[0] == next_day_high_values_normalised.shape[0] == next_day_low_values_normalised.shape[0] == next_day_close_values_normalised.shape[0] == next_day_volume_values_normalised.shape[0] == technical_indicators_normalised.shape[0]
    return ohlcv_histories_normalised, technical_indicators_normalised, next_day_open_values_normalised, next_day_open_values, y_normaliser_open, next_day_high_values_normalised, next_day_high_values, y_normaliser_high, next_day_low_values_normalised, next_day_low_values, y_normaliser_low, next_day_close_values_normalised, next_day_close_values, y_normaliser_close, next_day_volume_values_normalised, next_day_volume_values, y_normaliser_volume


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
