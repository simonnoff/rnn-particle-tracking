import math
import random
import matplotlib.pyplot as plt

# 1. Generate one particle positioned randomly on 2D grid
# 2. Generate a sequence of n steps which will move the particle in random directions

grid_height = 100
grid_width = 100

number_of_random_steps = 5
number_of_all_steps = 50
next_particle_radius = 10

color_code_init_point = '#FF0000'
color_code_next_points = '#0000FF'

points_dict = {}

def generate_random_particle():
    x = round(random.uniform(0, grid_width), 2)
    y = round(random.uniform(0, grid_height), 2)
    update_points_dict(0, x, y, 0, 0)

    return x, y

def generate_next_particle_randomly(index, previous_x, previous_y):
    x_lower_bound = previous_x - next_particle_radius
    x_lower_bound = x_lower_bound if x_lower_bound > 0 else 0
    x_upper_bound = previous_x + next_particle_radius
    x_upper_bound = x_upper_bound if x_upper_bound < grid_width else grid_width

    y_lower_bound = previous_y - next_particle_radius
    y_lower_bound = y_lower_bound if y_lower_bound > 0 else 0
    y_upper_bound = previous_y + next_particle_radius
    y_upper_bound = y_upper_bound if y_upper_bound < grid_height else grid_height

    next_particle_x = round(random.uniform(x_lower_bound, x_upper_bound), 2)
    next_particle_y = round(random.uniform(y_lower_bound, y_upper_bound), 2)

    # TODO: calculate direction and length based on previous x, previous y, next x and next y
    new_length = ((next_particle_x - previous_x)**2 + (next_particle_y - previous_y)**2)**(1/2)
    new_direction = math.degrees(math.atan2(next_particle_y - previous_y, next_particle_x - previous_x))

    update_points_dict(index, next_particle_x, next_particle_y, new_direction, new_length)

    return next_particle_x, next_particle_y

def generate_next_particle_sequentially(index, previous_x, previous_y):

    d_sum = 0
    l_sum = 0
    total_points = 1

    for i, point in points_dict.items():
        d = point["direction"]
        l = point["length"]

        d_sum += d
        l_sum += l
        total_points += 1

    d_avr = d_sum / total_points
    l_avr = l_sum / total_points

    new_direction = d_avr * (total_points % 2 + 1)
    new_length = l_avr * 2/(total_points % 2 + 1)

    new_direction = new_direction if new_direction < 360 else 45

    # TODO: calculate x and y based on new direction, new length, previous x and previous y
    x_difference = math.cos(new_direction) * new_length
    y_difference = math.sin(new_direction) * new_length
    new_x = x_difference + previous_x
    new_y = y_difference + previous_y

    new_x = new_x if new_x >= 0 and new_x <= grid_width else grid_width/2
    new_y = new_y if new_y >= 0 and new_y <= grid_height else grid_height / 2

    update_points_dict(index, new_x, new_y, new_direction, new_length)

    return new_x, new_y

def update_points_dict(index, x, y, direction, length):
    point = {
        "x": x,
        "y": y,
        "direction": direction,
        "length": length
    }

    points_dict[index] = point

def display_next_point(i, x, y):
    # Plot random particle
    color_code = color_code_init_point if i == 0 else color_code_next_points
    ax.scatter(x, y, color=color_code)
    ax.annotate(i, (x, y))
    plt.pause(0.1)

    print('X_'+ str(i), ': ', x, 'Y_' + str(i), ': ', y)

# Executable code

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

plt.axis([0, grid_width, 0, grid_height])

x_generated, y_generated = generate_random_particle()

for i in range(number_of_all_steps):
    display_next_point(i, x_generated, y_generated)
    if i < number_of_random_steps:
        x_generated, y_generated = generate_next_particle_randomly(i+1, x_generated, y_generated)
    else:
        x_generated, y_generated = generate_next_particle_sequentially(i+1, x_generated, y_generated)

plt.show()

# Having the initial point, plot the expected and predicted set of next points.
# Randomly plotting the data won't build any sufficient model, so we need to create a trend with all data points.

# Read document from Iain: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2747604/pdf/nihms95285.pdf

# 1. First point has random coordinates
# 2. Every next point has direction (degrees offset to x axis) and length
# 3. Direction is calculated with f = (avg. of previous directions) * (num. of points mod 4)
# 4. Length is calculated with g = (avg. of previous lengths) * 2/(num. of points mod 4)

# Once you have the data use RNN with TensorFlow to predict the next dots in the sequence.


clear;
% Adjust this parameter to control the direction: very large values
% correspond to isotropic scattering; small values to preferential forward
% motion.
sigma = 1e6;
x=0;y=0
theta = 0;
xtrack = [x];
ytrack = [y];
ct=[]
g = 0;
for i=1:10000
     s = rand(1);
     x = x+s*cos(theta);
     y = y+s*sin(theta);
     if g==0
         ctnew = 1-2*rand(1);
     else
         ctnew = (1+g^2 -( (1-g^2) / (1-g+2*g*rand(1)) )^2)/(2*g);
     end
     t = (2*randn(1)*sigma)*pi;
     theta = theta + t ;
     xtrack = [xtrack,x];
     ytrack = [ytrack,y];
end

plot(xtrack,ytrack)



