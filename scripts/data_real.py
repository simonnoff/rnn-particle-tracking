import common
import matplotlib.pyplot as plt
import data_helper as dh

number_of_points_per_frame = common.number_of_points_per_frame
movie_number = common.movie_number
display_points = False

helper = dh.DataHelper()

def file_name(movie_number, axis):
    return "../real_data/TC%s-C2_gui2_steps_%s.txt" % (movie_number, axis)

def run():
    file_X = open(file_name(common.movie_number, 'X'), 'r')
    file_Y = open(file_name(common.movie_number, 'Y'), 'r')

    X_lines = file_X.readlines()
    Y_lines = file_Y.readlines()

    for index in range(number_of_points_per_frame):
        elements_X = X_lines[index].split(',')
        elements_Y = Y_lines[index].split(',')

        last_e_X = '0'
        last_e_Y = '0'

        for el_index in range(len(elements_X)):
            e_X = elements_X[el_index]
            e_Y = elements_Y[el_index]

            if e_X == 'NaN' or e_Y == 'NaN':
                e_X = last_e_X
                e_Y = last_e_Y

            last_e_X = e_X
            last_e_Y = e_Y

            current_X = float(e_X)
            current_Y = float(e_Y)

            colors_array = ['#3300FF', '#660033',
                        '#333300', '#999933', '#FFCC33', '#CC6600', '#FF6600', '#CC3333', '#993333', '#996666',
                        '#000000'] * int(number_of_points_per_frame / 10)
            if display_points:
                plt.scatter(current_X, current_Y, s=10, color=colors_array[index])

            final_point = ([current_X, current_Y], index)
            helper.set_point_element(el_index, index, final_point)

    helper.save_data()
    helper.clear_points()

run()

if display_points:
    plt.show()
