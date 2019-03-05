import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import common
import transform_data
import model as md

# Read data
data = pd.read_pickle(common.data_path_pickle)
print("Data shape:", data.shape)

# Reshape data
input_data, output_data = transform_data.reshape_data_correctly(data)

# Split data
train_X, validation_X, test_X, train_y, validation_y, test_y = transform_data.split_data(input_data, output_data, train_ratio=0.8)
print("Train shape: %s, Val shape: %s, Test shape: %s" % (train_X.shape, validation_X.shape, test_X.shape))

# Define model
model = md.ConvLSTMModel()

# Train model
model.train(train_X, train_y, validation_X, validation_y)

# Test model
model.test(test_X, test_y)
