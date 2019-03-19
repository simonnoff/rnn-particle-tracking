import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import math
import common

'''
Reshape data, so we can train and test the model
'''
frame_window = 100

def calculate_window_size(frames):
    final = frame_window
    while frames % final != 0:
        final -= 1
    return final

def reshape_data_correctly(data):
    num_frames = int((data.shape[1]) / (common.point_coordinates + common.number_of_classes))

    frame_window_size = calculate_window_size(num_frames)
    num_samples = int(num_frames / frame_window_size)

    # Assign num_samples and frame_window_size in common.py
    common.number_of_samples = num_samples
    common.frame_window_size = frame_window_size

    number_of_point_coordinates_data = common.point_coordinates * num_frames
    number_of_probabilities_data = common.number_of_classes * num_frames

    input_shape = (num_samples, frame_window_size, common.number_of_points_per_frame, common.point_coordinates)
    output_shape = (num_samples, frame_window_size, common.number_of_points_per_frame, common.number_of_classes)

    input_data_array = np.zeros(shape=input_shape)
    output_data_array = np.zeros(shape=output_shape)

    print(output_shape)

    print("NPCD", number_of_point_coordinates_data)

    for index, point in enumerate(data.values):

        current_frame_output_num = 0
        current_window_step = 0
        current_sample_number = 0
        current_sample_number_output = 0
        current_point_index = 0

        for i, value in enumerate(point):

            if i < number_of_point_coordinates_data:

                current_point_index = i % common.point_coordinates

                if current_point_index == 0 and i > 0:
                    if current_window_step < frame_window_size - 1:
                        current_window_step += 1
                    else:
                        current_window_step = 0
                        current_sample_number += 1

                input_data_array[current_sample_number, current_window_step, index, current_point_index] = value

            else:

                current_point_index = i % common.number_of_classes

                if i == number_of_point_coordinates_data:
                    current_window_step = 0
                    current_sample_number = 0
                elif current_point_index == 0 and i > number_of_point_coordinates_data:
                    if current_window_step < frame_window_size - 1:
                        current_window_step += 1
                    else:
                        current_window_step = 0
                        current_sample_number += 1

                if current_sample_number != num_samples:
                    output_data_array[current_sample_number, current_window_step, index, current_point_index] = value

    return np.expand_dims(input_data_array, axis=4), output_data_array

'''
Split data into train/validation/test with ratio 80%/10%/10%
'''
def split_data(input_data, output_data, train_ratio = 0.8):
    train_split_ratio = train_ratio
    val_split_ratio = (1 - train_ratio) / 2

    data_length = len(input_data)
    n_train = math.ceil(train_split_ratio * data_length)
    n_val = math.ceil(val_split_ratio * data_length)

    train_X = input_data[:n_train, :]
    validation_X = input_data[n_train:(n_train + n_val), :]
    test_X = input_data[(n_train + n_val):, :]

    train_y = output_data[:n_train, :]
    validation_y = output_data[n_train:(n_train + n_val), :]
    test_y = output_data[(n_train + n_val):, :]

    return train_X, validation_X, test_X, train_y, validation_y, test_y
