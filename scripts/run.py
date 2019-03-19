import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np

import common
import transform_data
import conv_lstm_model as clm
import baseline_model as bm

should_train = common.should_train
use_new_data = common.use_new_data
split_data = common.split_data

# Read data
data = pd.read_pickle(common.data_path_pickle)
print("Data shape:", data.shape)

# Reshape data
input_data, output_data = transform_data.reshape_data_correctly(data)

if not use_new_data or split_data:
    # Split data
    train_X, validation_X, test_X, train_y, validation_y, test_y = transform_data.split_data(input_data, output_data, train_ratio=0.8)
    print("Train shape: %s, Val shape: %s, Test shape: %s" % (train_X.shape, validation_X.shape, test_X.shape))

# Define model
conv_lstm_model = clm.ConvLSTMModel()
baseline_model = bm.BaselineModel()

# Train model
if should_train and not use_new_data:
    conv_lstm_model.train(train_X, train_y, validation_X, validation_y)

# Test model
if use_new_data:
    conv_lstm_model.test(input_data, output_data)
    baseline_model.test(input_data, output_data)
else:
    conv_lstm_model.test(test_X, test_y)
    #baseline_model.test(test_X, test_y)
    conv_lstm_model.visualize(input_data, test_X)