import pandas as pd
from sklearn import preprocessing
import numpy as np
from scipy import signal

history_points = 50
N=200+1


def csv_to_dataset(csv_path):
    data = pd.read_csv(csv_path)
    data = data.drop('date', axis=1)
    data = data.drop(0, axis=0)

    data = data.values
    # print(data[:][:10])
    print(data.shape)
    
    d=data[:, 0].copy()
    d=50-1*d
    d=np.absolute(d)
    d_op = np.expand_dims(d[:], -1)

    d1=data[:, 0].copy()
    d1=100-1*d1
    d1=np.absolute(d1)
    d1_op = np.expand_dims(d1[:], -1)
    
    d2=data[:, 0].copy()
    d2=200-1*d2
    d2=np.absolute(d2)
    d2_op = np.expand_dims(d2[:], -1)
    
    d3=data[:, 0].copy()
    d3=400-1*d3
    d3=np.absolute(d3)
    # d3_=d3*np.random.rand(d3.shape[0])
    d3_op = np.expand_dims(d3[:], -1)
     
    dd=np.array([d[len(d)-i-1].copy() for i in range(len(d))])
    dd_op = np.expand_dims(dd[:], -1)
    
    dd1=np.array([d1[len(d1)-i-1].copy() for i in range(len(d1))])
    dd1_op = np.expand_dims(dd1[:], -1)
    
    dd2=np.array([d2[len(d2)-i-1].copy() for i in range(len(d2))])
    dd2_op = np.expand_dims(dd2[:], -1)
    
    dd3=np.array([d3[len(d3)-i-1].copy() for i in range(len(d3))])
    dd3_op = np.expand_dims(dd3[:], -1)
    
    y_normaliser = preprocessing.MinMaxScaler()
    data_ = np.concatenate((data[:,0], data[:,1], data[:,2],data[:,3],d,d1,d2,d3,dd,dd1,dd2,dd3))
    data_ = np.expand_dims(data_, -1)
    y_normaliser.fit(data_)
    
    

    
    dat0=data[:, 0].copy()
    dat0_op = np.expand_dims(dat0[:], -1)
    dat0_normalised=y_normaliser.transform(dat0_op)
    
    dat1=data[:, 1].copy()
    dat1_op = np.expand_dims(dat1[:], -1)
    dat1_normalised=y_normaliser.transform(dat1_op)
    
    dat2=data[:, 2].copy()
    dat2_op = np.expand_dims(dat2[:], -1)
    dat2_normalised=y_normaliser.transform(dat2_op)
    
    dat3=data[:, 3].copy()
    dat3_op = np.expand_dims(dat3[:], -1)
    dat3_normalised=y_normaliser.transform(dat3_op)


    d_normalised=y_normaliser.transform(d_op)
    d1_normalised=y_normaliser.transform(d1_op)
    d2_normalised=y_normaliser.transform(d2_op)
    d3_normalised=y_normaliser.transform(d3_op)
     
    dd_normalised=y_normaliser.transform(dd_op)
    dd1_normalised=y_normaliser.transform(dd1_op)
    dd2_normalised=y_normaliser.transform(dd2_op)
    dd3_normalised=y_normaliser.transform(dd3_op)
    
    data_pr_normalised = np.append(dat0_normalised,dat1_normalised,axis=1)
    data_pr_normalised = np.append(data_pr_normalised,dat2_normalised,axis=1)
    data_pr_normalised = np.append(data_pr_normalised,dat3_normalised,axis=1)
    
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
    
    dd_normalised1 = np.append(dd_normalised,dd1_normalised,axis=1)
    dd_normalised1 = np.append(dd_normalised1,dd2_normalised,axis=1)
    dd_normalised1 = np.append(dd_normalised1,dd3_normalised,axis=1)
    
    data_normalised1 = np.append(dd_normalised1,data_vl_normalised,axis=1)
    
    
    
    
    x=np.linspace(-np.pi*1, np.pi*1, data[:, 0].shape[0])
    y=np.sin(x)*150
    y=np.absolute(y)
    x1=np.linspace(-np.pi*2, np.pi*2, data[:, 0].shape[0])
    y1=np.sin(x1)*150
    y1=np.absolute(y1)
    x2=np.linspace(-np.pi*3, np.pi*3, data[:, 0].shape[0])
    y2=np.sin(x2)*150
    y2=np.absolute(y2)
    x3=np.linspace(-np.pi*4, np.pi*4, data[:, 0].shape[0])
    y3=np.sin(x3)*150
    y3=np.absolute(y3)
    x4=np.linspace(-np.pi*5, np.pi*5, data[:, 0].shape[0])
    y4=np.sin(x4)*150
    y4=np.absolute(y4)
    day=y.copy()
    day1=y1.copy()
    day2=y2.copy()
    day3=y3.copy()
    day4=y4.copy()
    
    y_normaliser = preprocessing.MinMaxScaler()
    data_ = np.concatenate((day, day1, day2, day2, day4))
    data_ = np.expand_dims(data_, -1)
    y_normaliser.fit(data_)
    
    day = np.expand_dims(day[:], -1)
    day_normalised=y_normaliser.transform(day)
    
    day1 = np.expand_dims(day1[:], -1)
    day1_normalised=y_normaliser.transform(day1)
    
    day2 = np.expand_dims(day2[:], -1)
    day2_normalised=y_normaliser.transform(day2)
    
    day3 = np.expand_dims(day3[:], -1)
    day3_normalised=y_normaliser.transform(day3)
    
    day4 = np.expand_dims(day4[:], -1)
    day4_normalised=y_normaliser.transform(day4)
    
    data_normalised2 = np.append(day_normalised,day1_normalised,axis=1)
    data_normalised2 = np.append(data_normalised2,day2_normalised,axis=1)
    data_normalised2 = np.append(data_normalised2,day3_normalised,axis=1)
    data_normalised2 = np.append(data_normalised2,day4_normalised,axis=1)
    
    # x=np.linspace(-np.pi*(0.5), np.pi*(0.5), data[:, 0].shape[0])
    # y=np.sin(x+np.pi/4)*150
    # y=np.absolute(y)

    # t = np.linspace(0, 1, data[:, 0].shape[0], endpoint=False)
    # sig = np.sin(2 * np.pi * t)
    # dty =(sig + 1)/2
    # pwm = signal.square(2 * np.pi * 2 * t, duty=dty)+1
    # y=pwm*150/2
    
    next_day=y.copy()


    # next_day=data[:, 0].copy()
    next_d=np.array([next_day[i + 0 + 0].copy() for i in range(len(next_day) - history_points - N)])
    next_d = np.expand_dims(next_d[:], -1)
    next_d_normalised=y_normaliser.transform(next_d)
    
    next_d1=np.array([next_day[i + history_points + -1].copy() for i in range(len(next_day) - history_points - N)])
    next_d1 = np.expand_dims(next_d1[:], -1)
    next_d1_normalised=y_normaliser.transform(next_d1)
    
    next_d2=np.array([next_day[i + history_points + 100].copy() for i in range(len(next_day) - history_points - N)])
    next_d2 = np.expand_dims(next_d2[:], -1)
    next_d2_normalised=y_normaliser.transform(next_d2)
    
    next_d3=np.array([next_day[i + history_points + 200].copy() for i in range(len(next_day) - history_points - N)])
    next_d3 = np.expand_dims(next_d3[:], -1)
    next_d3_normalised=y_normaliser.transform(next_d3)
    
    next_day_open_values=next_d
    next_day_open_values_normalised=next_d_normalised
    
    next_day_open_values1=next_d1
    next_day_open_values_normalised1=next_d1_normalised
    
    next_day_open_values2=next_d2
    next_day_open_values_normalised2=next_d2_normalised
    
    next_day_open_values3=next_d3
    next_day_open_values_normalised3=next_d3_normalised

    # y=np.sin(x+np.pi*1/2)*150
    # y=np.absolute(y)
    
    # t = np.linspace(0, 1, data[:, 0].shape[0], endpoint=False)
    # sig = np.sin(2 * np.pi * t)
    # dty =(sig + 1)/2
    # pwm = signal.square(2 * np.pi * 2 * t, duty=dty)+1
    # y=pwm*150/2
    
    # t = np.linspace(0, 1, data[:, 0].shape[0], endpoint=False)
    # pwm = signal.square(np.pi * 4 * t-np.pi/2)+1
    # y=pwm*150/2
    
    # x=np.linspace(-np.pi*(0.5), np.pi*(0.5), data[:, 0].shape[0])
    # y=np.sin(x+np.pi/4)*150
    # y=np.absolute(y)
    
    next_day=y.copy()

    next_day=np.expand_dims(next_day[:], -1)
    next_day_normalised=y_normaliser.transform(next_day)
    # next_day_normalised=day_normalised
    data_normalised3 = np.append(next_day_normalised,next_day_normalised,axis=1)
    data_normalised3 = np.append(data_normalised3,next_day_normalised,axis=1)
    data_normalised3 = np.append(data_normalised3,next_day_normalised,axis=1)
    data_normalised3 = np.append(data_normalised3,next_day_normalised,axis=1)

    

    # using the last {history_points} open close high low volume data points, predict the next open value
    # ohlcv_histories_normalised1 = np.array([data_normalised[i:i + history_points].copy() for i in range(len(data_normalised) - history_points - N)])
    # ohlcv_histories_normalised = np.array([data_normalised1[i:i + history_points].copy() for i in range(len(data_normalised1) - history_points - N)])
    ohlcv_histories_normalised1 = np.array([data_normalised3[i:i + history_points].copy() for i in range(len(data_normalised3) - history_points - N)])
    ohlcv_histories_normalised = np.array([data_normalised2[i:i + history_points].copy() for i in range(len(data_normalised2) - history_points - N)])
    ohlcv_histories_normalised2=np.array([(np.array([ohlcv_histories_normalised[i,:,:].copy()[len(ohlcv_histories_normalised[i,:,:].copy())-j-1].copy() for j in range(len(ohlcv_histories_normalised[i,:,:].copy()))])).copy() for i in range(len(ohlcv_histories_normalised))])

    # ohlcv_histories_normalised=ohlcv_histories_normalised2.copy()

    # next_day_open_values_normalised = np.array([data_normalised[:, 0][i + history_points + 30].copy() for i in range(len(data_normalised) - history_points - N)])
    # next_day_open_values_normalised = np.expand_dims(next_day_open_values_normalised, -1)
    # next_day_open_values = np.array([data[:, 0][i + history_points + 30].copy() for i in range(len(data) - history_points - N)])
    # next_day_open_values = np.expand_dims(next_day_open_values, -1)

    # next_day_open_values_normalised1 = np.array([data_normalised[:, 0][i + history_points + 20].copy() for i in range(len(data_normalised) - history_points - N)])
    # next_day_open_values_normalised1 = np.expand_dims(next_day_open_values_normalised1, -1)
    # next_day_open_values1 = np.array([data[:, 0][i + history_points + 20].copy() for i in range(len(data) - history_points - N)])
    # next_day_open_values1 = np.expand_dims(next_day_open_values1, -1)

    # next_day_open_values_normalised2 = np.array([data_normalised[:, 0][i + history_points + 10].copy() for i in range(len(data_normalised) - history_points - N)])
    # next_day_open_values_normalised2 = np.expand_dims(next_day_open_values_normalised2, -1)
    # next_day_open_values2 = np.array([data[:, 0][i + history_points + 10].copy() for i in range(len(data) - history_points - N)])
    # next_day_open_values2 = np.expand_dims(next_day_open_values2, -1)        

    # next_day_open_values_normalised3 = np.array([data_normalised[:, 0][i + history_points + -1].copy() for i in range(len(data_normalised) - history_points - N)])
    # next_day_open_values_normalised3 = np.expand_dims(next_day_open_values_normalised3, -1)
    # next_day_open_values3 = np.array([data[:, 0][i + history_points + -1].copy() for i in range(len(data) - history_points - N)])
    # next_day_open_values3 = np.expand_dims(next_day_open_values3, -1)
    
    # next_day_open_values_normalised3 = np.array([d3_normalised[:, 0][i + history_points + -1].copy() for i in range(len(d3_normalised) - history_points - N)])
    # next_day_open_values_normalised3 = np.expand_dims(next_day_open_values_normalised3, -1)
    # next_day_open_values3 = np.array([d3_op[:, 0][i + history_points + -1].copy() for i in range(len(d3_op) - history_points - N)])
    # next_day_open_values3 = np.expand_dims(next_day_open_values3, -1)

    next_day_open_values_normalised4 = np.array([data_normalised[:, 0][i + history_points + 5].copy() for i in range(len(data_normalised) - history_points - N)])
    next_day_open_values_normalised4 = np.expand_dims(next_day_open_values_normalised4, -1)
    next_day_open_values4 = np.array([data[:, 0][i + history_points + 5].copy() for i in range(len(data) - history_points - N)])
    next_day_open_values4 = np.expand_dims(next_day_open_values4, -1)

    # y_normaliser = data_pr_normaliser
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
    for his in ohlcv_histories_normalised1:
        # note since we are using his[3] we are taking the SMA of the closing price
        # sma = np.mean(his[:, 3])
        sma = his[49,3] # sma = his[49,3] # for make shape like Y 
        macd = calc_ema(his, 12) - calc_ema(his, 26)
        #technical_indicators.append(np.array([sma]))
        technical_indicators.append(np.array([sma,macd]))

    technical_indicators_normalised = np.array(technical_indicators)

    # tech_ind_scaler = preprocessing.MinMaxScaler()
    # technical_indicators_normalised = tech_ind_scaler.fit_transform(technical_indicators)

    assert ohlcv_histories_normalised.shape[0]  == technical_indicators_normalised.shape[0] == next_day_open_values_normalised.shape[0] == next_day_open_values_normalised1.shape[0] == next_day_open_values_normalised2.shape[0] == next_day_open_values_normalised3.shape[0] == next_day_open_values_normalised4.shape[0]
    return ohlcv_histories_normalised, technical_indicators_normalised,y_normaliser, next_day_open_values_normalised, next_day_open_values, next_day_open_values_normalised1, next_day_open_values1, next_day_open_values_normalised2, next_day_open_values2, next_day_open_values_normalised3, next_day_open_values3, next_day_open_values_normalised4, next_day_open_values4


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
