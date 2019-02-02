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

a = [([0, 0], 0), ([4, 1], 1), ([1, 3], 2), ([4, 1], 3), ([10, 0.5], 4)]

sorted_array = sort_point_per_frame(a)

print("Sorted array:", sorted_array)