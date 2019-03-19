import os
import numpy as np

# CAN CHANGE
number_of_points_per_frame = number_of_classes = 20
number_of_time_steps = 10000 # for the real
sigma = 10.0 # or 1e6 # increase sigma
upper_sigma = 10.0
use_new_data = False
should_train = True
split_data = True

is_generated = True
movie_number = 641

# DO NOT CHANGE
point_coordinates = 2
frame_window_size = 0
number_of_samples = 0

def sigma_to_string(s):
    first = int(s)
    second = str(s-first)[2:]
    return str(first) + "_" + second

generated_data = "../data/generated_points_%s_time_%s_sigma_%s_bounding_box%s.pkl" % (number_of_points_per_frame, number_of_time_steps, sigma_to_string(sigma), '_fresh' if use_new_data else '')
real_data = "../data/real_movie_%s_points_%s_time_%s.pkl" % (movie_number, number_of_points_per_frame, number_of_time_steps)

data_path_pickle = generated_data if is_generated else real_data

checkpoint_path = os.path.dirname(os.path.dirname( __file__ )) + "/training_checkpoint/model_points_%s_time_%s_sigma_%s.ckpt" % (number_of_points_per_frame, number_of_time_steps, sigma_to_string(sigma))

def evaluation_metric(accuracy_matrix):
    result = np.sum(accuracy_matrix, axis=1)
    result = result / accuracy_matrix.shape[1]
    final_value = np.average(result)
    return final_value