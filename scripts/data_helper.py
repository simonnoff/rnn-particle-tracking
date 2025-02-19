import common
import math
import datetime, time
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

    def _current_time(self):
        return datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
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

    def _prob_single(self, next_arr, curr_ele):
        index = np.where(next_arr == curr_ele[1])[0][0]
        empty_arr = np.zeros(number_of_classes)
        empty_arr[index] = 1
        return empty_arr

    def _generate_probability_matrices(self, points_array, final_points):
        previous_points_subarray = None

        for i, current_points_subarray in enumerate(points_array):
            curr_points_indicies = list(map(lambda x: x[1], current_points_subarray))
            curr_points_coordinates = map(lambda x: x[0], current_points_subarray)

            for point_index, point in enumerate(curr_points_coordinates):
                new_index = i * 2
                final_points[point_index][new_index] = point[0]
                final_points[point_index][new_index + 1] = point[1]

            if i == 0:
                previous_points_subarray = current_points_subarray
                continue

            self.probability_matrices[i] = list(map(lambda x: self._prob_single(curr_points_indicies, x), previous_points_subarray))

            previous_points_subarray = current_points_subarray

            # for j in range(number_of_points_per_frame):
            #     point = current_points_subarray[j]
            #     point_index = point[1]
            #     next_array_point_index = [i for i, a in enumerate(next_points_subarray) if a[1] == point_index][0]
            #     self.probability_matrices[i][j][next_array_point_index] = 1
            if i % 100 == 0:
                print("Here is fast 1.1 with index %s and time %s" % (i, self._current_time()))

    def save_data(self):
        print("Start generating points...", self._current_time())

        correctly_sorted_array = map(self._sort_point_per_frame, self.points) # efficient - SLOW

        double_time_steps = number_of_time_steps * 2
        final_points = np.zeros((number_of_points_per_frame, double_time_steps))

        print("Here is fast 1", self._current_time())
        self._generate_probability_matrices(correctly_sorted_array, final_points) # SLOW

        print("Here is fast 2", self._current_time())

        columns_time_steps = [("x_%s" % int(i / 2)) if i % 2 == 0 and i < double_time_steps else ("y_%s" % int((i - 1) / 2)) for i in
                range(double_time_steps)] # efficient
        columns_result = ["result_%s_t_%s" % (i % number_of_classes, math.floor(i / number_of_classes)) for i in range(number_of_time_steps * number_of_classes)] # efficient

        print("Here is fast 3", self._current_time())

        correct_probability_matrices = [[self.probability_matrices[j][i] for j in range(number_of_time_steps)] for i in
                                range(number_of_points_per_frame)]

        print("Here is fast 4", self._current_time())
        print("Random element", final_points[16])
        #final_points = np.array(correctly_sorted_array).reshape((number_of_points_per_frame, double_time_steps))
        final_probabilities = np.array(correct_probability_matrices).reshape((number_of_points_per_frame, number_of_time_steps * number_of_classes))

        print("Here is fast 5", self._current_time())
        columns = columns_time_steps + columns_result
        data_frame = pd.DataFrame(np.hstack((final_points, final_probabilities)), columns=columns)
        print("Here is fast 6", self._current_time())
        data_frame.to_pickle(data_path_pickle)

        print("Saved successfully", self._current_time())

    def set_point_element(self, axis1, axis2, result):
        self.points[axis1][axis2] = result

    def clear_points(self):
        self.points = np.zeros((number_of_time_steps, number_of_points_per_frame), dtype='2f, i')