import sys
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import common
import data_helper as dh

number_of_points_per_frame = common.number_of_points_per_frame
number_of_time_steps = common.number_of_time_steps
sigma = common.sigma
upper_sigma = common.upper_sigma

bounding_box_size = 200
display_points = False

helper = dh.DataHelper()

def determine_coordinates_bounding_box(coordinate_number):
    while coordinate_number > bounding_box_size or coordinate_number < -bounding_box_size:
        if coordinate_number > bounding_box_size:
            diff = coordinate_number - bounding_box_size # always positive
            coordinate_number = bounding_box_size - diff
        if coordinate_number < -bounding_box_size:
            diff = coordinate_number + bounding_box_size # always negative
            coordinate_number = - (bounding_box_size + diff)

    return coordinate_number

def generate_point_track(index):
    # Parameters for the random particle generation
    theta = 0
    range_bound = 10
    sigma = random.uniform(0, upper_sigma)
    x_generated, y_generated = random.uniform(-range_bound, range_bound), random.uniform(-range_bound, range_bound) # generate_random_particle()
    # Algorithm is based on https://en.wikipedia.org/wiki/Monte_Carlo_method_for_photon_transport?
    for i in range(number_of_time_steps):
        colors_array = ['#3300FF', '#660033',
                        '#333300', '#999933', '#FFCC33', '#CC6600', '#FF6600', '#CC3333', '#993333', '#996666',
                        '#000000'] * int(number_of_points_per_frame / 10)

        #print("Scatter index %s at time step %s" % (index, i))
        if display_points:
            plt.scatter(x_generated, y_generated, s=10, color=colors_array[index])
        # Eliminate 0 values for coordinates
        x_generated = 0.00001 if x_generated == 0 else x_generated
        y_generated = 0.00001 if y_generated == 0 else y_generated

        x_generated = determine_coordinates_bounding_box(x_generated)
        y_generated = determine_coordinates_bounding_box(y_generated)

        point = ([x_generated, y_generated], index)
        helper.set_point_element(i, index, point)

        # random.uniform(0, 1) - sample from range distribution
        s = random.uniform(0, 1)

        x_generated = x_generated + s * math.cos(theta)
        y_generated = y_generated + s * math.sin(theta)

        # np.random.randn() - sample from the standard distribution
        t = (2*np.random.randn()*sigma)*math.pi
        theta = theta + t

for i in range(number_of_points_per_frame):
    generate_point_track(i)

helper.save_data()

if display_points:
    plt.show()
