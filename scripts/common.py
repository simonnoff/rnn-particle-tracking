# CAN CHANGE
number_of_points_per_frame = number_of_classes = 5
number_of_time_steps = 1000
sigma = 20.0 # or 1e6 # increase sigma

# DO NOT CHANGE
point_coordinates = 2
frame_window_size = 0

def sigma_to_string(s):
    first = int(s)
    second = str(s-first)[2:]
    return str(first) + "_" + second

data_path_pickle = "../data/generated_points_%s_time_%s_sigma_%s.pkl" % (number_of_points_per_frame, number_of_time_steps, sigma_to_string(sigma))
