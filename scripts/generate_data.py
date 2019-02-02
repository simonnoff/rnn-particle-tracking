import sys
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

data_path = "../data/generated_points_final_format.csv"

color_code_init_point = '#FF0000'
color_code_next_points = '#0000FF'

number_of_points_per_frame = number_of_classes = 3
number_of_time_steps = 5

points = probability_matrices = [[[] for i in range(number_of_points_per_frame)] for j in range(number_of_time_steps)]

'''
Sort points based on closest to the bottom of the coordinate system
'''
def compare_two_points(point1, point2):
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

def sort_point_per_frame(array):
    # Quicksort
    equal = []
    less = []
    higher = []

    if len(array) > 1:
        pivot = array[0]
        for x in array:
            result = compare_two_points(x, pivot)
            if result == 'smaller':
                less.append(x)
            elif result == 'bigger':
                higher.append(x)
            else:
                equal.append(x)

        return sort_point_per_frame(less) + equal + sort_point_per_frame(higher)
    else:
        return array


def generate_point_track(index):
    # Parameters for the random particle generation
    sigma = 0.0001 # or 1e6
    theta = 0

    x_generated, y_generated = 0, 0 # generate_random_particle()
    # Algorithm is based on https://en.wikipedia.org/wiki/Monte_Carlo_method_for_photon_transport?
    for i in range(number_of_time_steps):
        # Eliminate 0 values for coordinates
        x_generated = 0.00001 if x_generated == 0 else x_generated
        y_generated = 0.00001 if y_generated == 0 else y_generated

        point = ([x_generated, y_generated], index)
        points[i][index] = point

        # random.uniform(0, 1) - sample from range distribution
        s = random.uniform(0, 1)

        x_generated = x_generated + s * math.cos(theta)
        y_generated = y_generated + s * math.sin(theta)

        # np.random.randn() - sample from the standard distribution
        t = (2*np.random.randn()*sigma)*math.pi
        theta = theta + t

def generate_probability_matrices(points_array):
    for i in range(number_of_time_steps-1):
        current_points_subarray = points_array[i]
        next_points_subarray = points_array[i+1]
        for j in range(number_of_points_per_frame):
            point = current_points_subarray[j]
            point_index = point[1]
            next_array_point_index = [i for i, a in enumerate(next_points_subarray) if a[1] == point_index][0]
            probability_matrices[i][j] = [0 if i != next_array_point_index else 1 for i in range(number_of_classes)]

        probability_matrices[number_of_time_steps-1] = [[0 for _ in range(number_of_classes)] for _ in range(number_of_points_per_frame)]


def save_data():
    print("Start generating points...")

    for i in range(number_of_points_per_frame):
        generate_point_track(i)

        if i == 100:
            print("Still generating...", i)

    correctly_sorted_array = [sort_point_per_frame(a) for a in points]
    generate_probability_matrices(correctly_sorted_array)

    print("Array's shape", np.array(correctly_sorted_array).shape)
    print("Sorted points\n", np.array(correctly_sorted_array))
    print("Probability matrices:\n", np.array(probability_matrices))

    double_time_steps = number_of_time_steps * 2

    columns_time_steps = [("x_%s" % int(i / 2)) if i % 2 == 0 and i < double_time_steps else ("y_%s" % int((i - 1) / 2)) for i in
               range(double_time_steps)]
    columns_result = ["result_%s_t_%s" % (i % number_of_classes, math.floor(i / number_of_classes)) for i in range(number_of_time_steps * number_of_classes)]

    correctly_sorted_array = [[correctly_sorted_array[j][i][0] for j in range(number_of_time_steps)] for i in range(number_of_points_per_frame)]
    correct_probability_matrices = [[probability_matrices[j][i] for j in range(number_of_time_steps)] for i in
                              range(number_of_points_per_frame)]

    final_points = np.array(correctly_sorted_array).reshape((number_of_points_per_frame, double_time_steps))
    final_probabilities = np.array(correct_probability_matrices).reshape((number_of_points_per_frame, number_of_time_steps * number_of_classes))

    print("Final points\n %s" % final_probabilities)
    data_frame = pd.DataFrame(np.hstack((final_points, final_probabilities)))
    data_frame.to_csv(data_path, header=columns_time_steps + columns_result)

arguments = sys.argv
if len(arguments) < 2 or arguments[1] not in ["save_data"]:
    print("> Please enter \"save_data\" parameter ")
    sys.exit(0)

if arguments[1] == "save_data":
    save_data()


