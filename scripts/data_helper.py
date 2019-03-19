import common
import math
import numpy as np
import pandas as pd

data_path_pickle = common.data_path_pickle
number_of_points_per_frame = number_of_classes = common.number_of_points_per_frame
number_of_time_steps = common.number_of_time_steps

class DataHelper:
    def __init__(self):
        self.data_path_pickle = common.data_path_pickle
        self.points = np.zeros((number_of_time_steps, number_of_points_per_frame), dtype='2f, i')
        self.probability_matrices = np.zeros((number_of_time_steps, number_of_points_per_frame, number_of_classes))

    '''
    Sort points based on closest to the bottom of the coordinate system
    '''
    def _compare_two_points(self, point1, point2):
        point1_x = point1[0][0]
        point1_y = point1[0][1]

        point2_x = point2[0][0]
        point2_y = point2[0][1]

        if point1_y < point2_y:
            return 'smaller'
        elif point1_y == point2_y:
            if point1_x < point2_x:
                return 'smaller'
            elif point1_x > point2_x:
                return 'bigger'
            else:
                return 'equal'
        elif point1_y > point2_y:
            return 'bigger'

    def _sort_point_per_frame(self, array):
        # Quicksort
        equal = []
        less = []
        higher = []

        if len(array) > 1:
            pivot = array[0]
            for x in array:
                result = self._compare_two_points(x, pivot)
                if result == 'smaller':
                    less.append(x)
                elif result == 'bigger':
                    higher.append(x)
                else:
                    equal.append(x)

            return self._sort_point_per_frame(less) + equal + self._sort_point_per_frame(higher)
        else:
            return array

    def _generate_probability_matrices(self, points_array):
        for i in range(number_of_time_steps-1):
            current_points_subarray = points_array[i]
            next_points_subarray = points_array[i+1]
            for j in range(number_of_points_per_frame):
                point = current_points_subarray[j]
                point_index = point[1]
                next_array_point_index = [i for i, a in enumerate(next_points_subarray) if a[1] == point_index][0]
                self.probability_matrices[i][j][next_array_point_index] = 1

    def save_data(self):
        print("Start generating points...")

        correctly_sorted_array = [self._sort_point_per_frame(a) for a in self.points]
        self._generate_probability_matrices(correctly_sorted_array)

        double_time_steps = number_of_time_steps * 2

        columns_time_steps = [("x_%s" % int(i / 2)) if i % 2 == 0 and i < double_time_steps else ("y_%s" % int((i - 1) / 2)) for i in
                range(double_time_steps)]
        columns_result = ["result_%s_t_%s" % (i % number_of_classes, math.floor(i / number_of_classes)) for i in range(number_of_time_steps * number_of_classes)]

        correctly_sorted_array = [[correctly_sorted_array[j][i][0] for j in range(number_of_time_steps)] for i in range(number_of_points_per_frame)]
        correct_probability_matrices = [[self.probability_matrices[j][i] for j in range(number_of_time_steps)] for i in
                                range(number_of_points_per_frame)]

        final_points = np.array(correctly_sorted_array).reshape((number_of_points_per_frame, double_time_steps))
        final_probabilities = np.array(correct_probability_matrices).reshape((number_of_points_per_frame, number_of_time_steps * number_of_classes))

        columns = columns_time_steps + columns_result
        data_frame = pd.DataFrame(np.hstack((final_points, final_probabilities)), columns=columns)
        data_frame.to_pickle(data_path_pickle)

        print("Saved successfully")

    def set_point_element(self, axis1, axis2, result):
        self.points[axis1][axis2] = result

    def clear_points(self):
        self.points = np.zeros((number_of_time_steps, number_of_points_per_frame), dtype='2f, i')