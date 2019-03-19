import common
import numpy as np

class BaselineModel():
    def __init__(self):
        pass

    def distance(self, starting_point, next_points_array):
        min_index_point = 0
        min_distance = -1
        for index, point in enumerate(next_points_array):
            distance = np.linalg.norm(starting_point - point)
            if distance < min_distance or min_distance < 0:
                min_distance = distance
                min_index_point = index

        return min_index_point


    def test(self, X, y):
        num_samples_X = X.shape[0]
        track_accuracy_matrix = np.zeros((common.number_of_points_per_frame, num_samples_X * common.frame_window_size))

        for sample_index, sample_X in enumerate(X):
            sample_y = y[sample_index]

            actual_indicies = np.argmax(sample_y, axis=2)

            for i in range(common.frame_window_size - 1):
                current_time_step = common.frame_window_size * sample_index + i
                current_points_array = sample_X[i]
                next_points_array = sample_X[i+1]

                for j in range(common.number_of_points_per_frame):
                    current_point = current_points_array[j]
                    next_point_index = self.distance(current_point, next_points_array)

                    track_accuracy_matrix[j][current_time_step] = 1 if next_point_index == actual_indicies[i][j] else 0

        print("[Baseline model] Final value:", common.evaluation_metric(track_accuracy_matrix))



