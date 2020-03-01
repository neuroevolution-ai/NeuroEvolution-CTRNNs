from svglib.svglib import svg2rlg
from reportlab.lib import colors


def read_neuron_coordinates_from_svg_file(filename):

    drawing = svg2rlg(filename)
    elements = drawing.contents[0].contents[0].contents

    coordinates = {'input_neurons': [], 'hidden_neurons': [], 'output_neurons': []}

    bounding_rectangle = None

    # Parse all elements in drawing to get rectangle
    for element in elements:

        # Check if current element's stroke color is black (only the rectangle must have this property)
        if element.strokeColor == colors.black:
            bounding_rectangle = element
            break

    if bounding_rectangle is not None:
        rect_x = bounding_rectangle.x
        rect_y = bounding_rectangle.y
        rect_width = bounding_rectangle.width
        rect_height = bounding_rectangle.height
    else:
        return None

    # Parse all elements in drawing to get coordinates of circles
    for element in elements:

        # Check if current element's stroke color is not black (all circles must not have this property)
        if element.strokeColor != colors.black:

            # Rescale coordinates of the circle
            x = (element.cx - rect_x) / rect_width
            y = (element.cy - rect_y) / rect_height

            # Input neurons are filled with red color
            if element.fillColor == colors.red:
                coordinates['input_neurons'].append([x, y])
            # Hidden neurons are filled with black color
            elif element.fillColor == colors.black:
                coordinates['hidden_neurons'].append([x, y])
            # Output neurons are filled with blue color
            elif element.fillColor == colors.blue:
                coordinates['output_neurons'].append([x, y])

    return coordinates


coords = read_neuron_coordinates_from_svg_file("Brain.svg")