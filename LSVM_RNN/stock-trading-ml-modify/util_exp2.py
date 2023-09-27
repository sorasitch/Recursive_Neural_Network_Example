import pandas as pd
from sklearn import preprocessing
import numpy as np

history_points = 50
N=5


def csv_to_dataset(csv_path):
    data = pd.read_csv(csv_path)
    data = data.drop('date', axis=1)
    data = data.drop(0, axis=0)

    data = data.values
    # print(data[:][:10])
    # print(data.shape)

    data_pr = np.concatenate((data[:,0], data[:,1], data[:,2],data[:,3]))
    # print(data_pr[:][:10])
    # print(data_pr.shape)
    data_pr = np.expand_dims(data_pr, -1)
    data_pr_normaliser = preprocessing.MinMaxScaler()
    # data_pr_normalised = data_normaliser.fit_transform(data_pr)
    # data_pr_normaliser = preprocessing.StandardScaler()
    # data_pr_normaliser.fit(data_pr)
    data_pr_normaliser.fit(data[:,:4])
    data_pr_normalised=data_pr_normaliser.transform(data[:,:4])
    # print(data_pr_normalised.shape)
    # print(data_pr_normalised[:][:10])

    data_vl = data[:,4]
    # print(data_vl[:][:10])
    # print(data_vl.shape)
    data_vl = np.expand_dims(data_vl, -1)
    data_vl_normaliser = preprocessing.MinMaxScaler()
    # data_vl_normalised = data_normaliser.fit_transform(data_vl)
    # data_vl_normaliser = preprocessing.StandardScaler()
    data_vl_normaliser.fit(data_vl)
    data_vl_normalised=data_vl_normaliser.transform(data_vl)
    # print(data_vl_normalised.shape)
    # print(data_vl_normalised[:][:10])

    data_normalised = np.append(data_pr_normalised,data_vl_normalised,axis=1)
    # print(data_normalised.shape)
    # print(data_normalised[:][:10])

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
    print(ohlcv_histories_normalised.shape)
    next_day_open_values_normalised = np.array([data_normalised[:, :][i + history_points + 0].copy() for i in range(len(data_normalised) - history_points - N)])
    print(next_day_open_values_normalised.shape)
    # next_day_open_values_normalised = np.expand_dims(next_day_open_values_normalised, -1)
    # print(next_day_open_values_normalised.shape)
    next_day_open_values = np.array([data[:,:][i + history_points + 0].copy() for i in range(len(data) - history_points - N)])
    print(next_day_open_values.shape)
    # next_day_open_values = np.expand_dims(next_day_open_values, -1)

    next_day_open_values_normalised1 = np.array([data_normalised[:, :][i + history_points + 1].copy() for i in range(len(data_normalised) - history_points - N)])
    # next_day_open_values_normalised1 = np.expand_dims(next_day_open_values_normalised1, -1)
    next_day_open_values1 = np.array([data[:, :][i + history_points + 1].copy() for i in range(len(data) - history_points - N)])
    # next_day_open_values1 = np.expand_dims(next_day_open_values1, -1)

    next_day_open_values_normalised2 = np.array([data_normalised[:, :][i + history_points + 2].copy() for i in range(len(data_normalised) - history_points - N)])
    # next_day_open_values_normalised2 = np.expand_dims(next_day_open_values_normalised2, -1)
    next_day_open_values2 = np.array([data[:, :][i + history_points + 2].copy() for i in range(len(data) - history_points - N)])
    # next_day_open_values2 = np.expand_dims(next_day_open_values2, -1)        

    next_day_open_values_normalised3 = np.array([data_normalised[:, :][i + history_points + 3].copy() for i in range(len(data_normalised) - history_points - N)])
    # next_day_open_values_normalised3 = np.expand_dims(next_day_open_values_normalised3, -1)
    next_day_open_values3 = np.array([data[:, :][i + history_points + 3].copy() for i in range(len(data) - history_points - N)])
    # next_day_open_values3 = np.expand_dims(next_day_open_values3, -1)

    next_day_open_values_normalised4 = np.array([data_normalised[:, :][i + history_points + 4].copy() for i in range(len(data_normalised) - history_points - N)])
    # next_day_open_values_normalised4 = np.expand_dims(next_day_open_values_normalised4, -1)
    next_day_open_values4 = np.array([data[:, :][i + history_points + 4].copy() for i in range(len(data) - history_points - N)])
    # next_day_open_values4 = np.expand_dims(next_day_open_values4, -1)

    y_normaliser = data_pr_normaliser
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

    tech_ind_scaler = preprocessing.MinMaxScaler()
    technical_indicators_normalised = tech_ind_scaler.fit_transform(technical_indicators)

    assert ohlcv_histories_normalised.shape[0]  == technical_indicators_normalised.shape[0] == next_day_open_values_normalised.shape[0] == next_day_open_values_normalised1.shape[0] == next_day_open_values_normalised2.shape[0] == next_day_open_values_normalised3.shape[0] == next_day_open_values_normalised4.shape[0]
    return ohlcv_histories_normalised, technical_indicators_normalised,y_normaliser,y_normaliser_volume, next_day_open_values_normalised, next_day_open_values, next_day_open_values_normalised1, next_day_open_values1, next_day_open_values_normalised2, next_day_open_values2, next_day_open_values_normalised3, next_day_open_values3, next_day_open_values_normalised4, next_day_open_values4


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
