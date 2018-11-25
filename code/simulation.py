import math
import random
import matplotlib.pyplot as plt
import numpy as np

# 1. Generate one particle positioned randomly on 2D grid
# 2. Generate a sequence of n steps which will move the particle in random directions

grid_height = 100
grid_width = 100

number_of_random_steps = 5
number_of_all_steps = 100
next_particle_radius = 10

color_code_init_point = '#FF0000'
color_code_next_points = '#0000FF'

points = []

def generate_random_particle():
    x = round(random.uniform(0, grid_width), 2)
    y = round(random.uniform(0, grid_height), 2)
    update_points_dict(0, x, y, 0, 0)
    
    return x, y

def display_next_point(i, x, y):
    # Plot random particle
    color_code = color_code_init_point if i == 0 else color_code_next_points
    ax.scatter(x, y, color=color_code)
    ax.annotate(i, (x, y))
    plt.pause(0.1)
    
    # print('X_'+ str(i), ': ', x, 'Y_' + str(i), ': ', y)

def generate_points():
    # Parameters for the random particle generation
    sigma = 1e6
    theta = 0
    g = random.uniform(-1, 1)
    
    x_generated, y_generated = 0, 0 # generate_random_particle()
    # Algorithm is based on https://en.wikipedia.org/wiki/Monte_Carlo_method_for_photon_transport?
    for i in range(number_of_all_steps):
        point = {"x": x_generated, "y": y_generated}
        points.append(point)
        
        # random.uniform(0, 1) - sample from range distribution
        s = random.uniform(0, 1)
        
        x_generated = x_generated + s * math.cos(theta)
        y_generated = y_generated + s * math.sin(theta)
        
        if g==0:
            ctnew = 1-2*random.uniform(0, 1)
        else:
            pow_g_2 = math.pow(g, 2)
            ctnew = (1 + pow_g_2 - math.pow((1 - pow_g_2) / (1 - g + 2*g*random.uniform(0, 1)), 2)) / (2*g)
        
        # np.random.randn() - sample from the standard distribution
        t = (2*np.random.randn()*sigma)*math.pi
        theta = theta + t

def display_points():
    #plt.axis([0, grid_width, 0, grid_height])

    for i, point in enumerate(points):
        display_next_point(i, point["x"], point["y"])

    plt.show()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

generate_points()
display_points()

# Having the initial point, plot the expected and predicted set of next points.
# Randomly plotting the data won't build any sufficient model, so we need to create a trend with all data points.

# Read document from Iain: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2747604/pdf/nihms95285.pdf

# Once you have the data use RNN with TensorFlow to predict the next dots in the sequence.



