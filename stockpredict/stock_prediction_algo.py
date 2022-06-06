from posixpath import split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from os.path import exists
import pickle
from django.conf import settings
from keras.models import load_model
scaler = MinMaxScaler(feature_range=(0, 1))


def scale_data(dataset, stock_name):
    # scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(np.array(dataset).reshape(-1, 1))
    return create_model(dataset, stock_name)


# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)


def create_model(df1, stock_name):
    file_exists = exists(settings.MEDIA_ROOT + stock_name + '.h5')

    time_step = 100
    training_size = int(len(df1)*0.65)
    test_size = len(df1)-training_size
    train_data, test_data = df1[0:training_size,
                                :], df1[training_size:len(df1), :1]
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, ytest = create_dataset(test_data, time_step)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    if(file_exists != True):
        # Create the Stacked LSTM model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
        model.add(LSTM(50, return_sequences=True))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(X_train, y_train, validation_data=(
            X_test, ytest), epochs=100, batch_size=64, verbose=1)
        model.save(
            settings.MEDIA_ROOT + stock_name + '.h5')
        return predict_future_prices(model, test_data, df1)
    else:
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
        model.add(LSTM(50, return_sequences=True))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.load_weights(settings.MEDIA_ROOT + stock_name + '.h5')
        # open(settings.MEDIA_ROOT + stock_name + '.sav', 'rb')
        return predict_future_prices(model, test_data, df1)


def predict_future_prices(model, test_data, df1):
    x_input = test_data[len(test_data) - 100:].reshape(1, -1)

    temp_input = list(x_input)
    temp_input = temp_input[0].tolist()

    lst_output = []
    n_steps = 100
    i = 0
    while(i < 10):
        if(len(temp_input) > 100):
            x_input = np.array(temp_input[1:])
            # print("{} day input {}".format(i, x_input))
            x_input = x_input.reshape(1, -1)
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            # print("{} day output {}".format(i, yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input = temp_input[1:]
            lst_output.extend(yhat.tolist())
            i = i+1
        else:
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            # print(yhat[0])
            temp_input.extend(yhat[0].tolist())
            # print(len(temp_input))
            lst_output.extend(yhat.tolist())
            i = i+1
    # df3 = df1.tolist()

    # df3.extend(lst_output)

    df3 = scaler.inverse_transform(lst_output).tolist()
    return df3
