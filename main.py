import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras.layers as layers
from math import sqrt
from keras.models import Sequential as seq
from sklearn.metrics import mean_squared_error as m_s_err

def split_dataset(data):
	train, test = data[0:168], data[168:241]
	train = np.array(np.split(train, len(train)/24))
	test = np.array(np.split(test, len(test)/24))
	return train, test

def show_plot(true, pred, title):
    #fig = plt.subplots()
    plt.plot(true, label='Y_original')
    plt.plot(pred, dashes=[4, 3], label='Y_predicted')
    plt.xlabel('N_samples', fontsize=12)
    plt.ylabel('Instance_value', fontsize=12)
    plt.title(title, fontsize=12)
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.show()

# evaluate one or more weekly forecasts against expected values
def evaluate_forecasts(actual, predicted):
	scores = list()
	# calculate an RMSE score for each day
	for i in range(actual.shape[1]):
		# calculate mse
		mse = m_s_err(actual[:, i], predicted[:, i])
		# calculate rmse
		rmse = sqrt(mse)
		# store
		scores.append(rmse)
	# calculate overall RMSE
	s = 0
	for row in range(actual.shape[0]):
		for col in range(actual.shape[1]):
			s += (actual[row, col] - predicted[row, col])**2
	
	show_plot(actual[1], predicted[1], "17.02.2023")
	score = sqrt(s / (actual.shape[0] * actual.shape[1]))
	return score, scores 

# convert history into inputs and outputs
def to_supervised(train, n_input, n_out=24):
	# flatten data
	data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
	x, y = list(), list()
	in_start = 0
	# step over the entire history one time step at a time
	for _ in range(len(data)):
		# define the end of the input sequence
		in_end = in_start + n_input
		out_end = in_end + n_out
		# ensure we have enough data for this instance
		if out_end <= len(data):
			x.append(data[in_start:in_end, :])
			y.append(data[in_end:out_end, 0])
		# move along one time step
		in_start += 1
	return np.array(x), np.array(y)

# train the model
def build_model(train, n_input):
	# prepare data
	train_x, train_y = to_supervised(train, n_input)
	# define parameters
	verbose, epochs, batch_size = 0, 50, 16
	n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
	# reshape output into [samples, timesteps, features]
	train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
	# define model
	model = seq()
	model.add(layers.LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
	model.add(layers.RepeatVector(n_outputs))
	model.add(layers.LSTM(200, activation='relu', return_sequences=True))
	model.add(layers.TimeDistributed(layers.Dense(100, activation='relu')))
	model.add(layers.TimeDistributed(layers.Dense(1)))
	model.compile(loss='mse', optimizer='adam')
	# fit network
	model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
	return model

# make a forecast
def forecast(model, history, n_input):
	# flatten data
	data = np.array(history)
	data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
	# retrieve last observations for input data
	input_x = data[-n_input:, :]
	# reshape into [1, n_input, n]
	input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
	# forecast the next week
	yhat = model.predict(input_x, verbose=0)
	# we only want the vector forecast
	yhat = yhat[0]
	return yhat

# evaluate a single model
def evaluate_model(train, test, n_input):
	# fit model
	model = build_model(train, n_input)
	# history is a list of weekly data
	history = [x for x in train]
	# walk-forward validation over each week
	predictions = list()
	for i in range(len(test)):
		# predict the week
		yhat_sequence = forecast(model, history, n_input)
		# store the predictions
		predictions.append(yhat_sequence)
		# get real observation and add to history for predicting the next week
		history.append(test[i, :])
	# evaluate predictions days for each week
	predictions = np.array(predictions)
	score, scores = evaluate_forecasts(test[:, :, 0], predictions)
	return score, scores

dataset = pd.read_csv('train.csv',header=0,infer_datetime_format=True, parse_dates=['datetime'],index_col=['datetime'])
train, test = split_dataset(dataset.values)
# train test
print(train.shape)
print(train[0, 0, 0], train[-1, -1, 0])
# validate test
print(test.shape)
print(test[0, 0, 0], test[-1, -1, 0])

n_input = 8
point, points = evaluate_model(train, test, n_input)


