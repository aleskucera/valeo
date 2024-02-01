import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Template(object):
    """Template is a class that represents a template for an object. It is a 2D array of
    -1s, 0s and 1s. The -1s represent the points outside the object, the 0s represent the
    points inside the object and the 1s represent the points on the boundary of the object.

    The template uses pytorch3d library to represent template as a point cloud. The points
    outside the object are colored green, the points inside the object are colored black and
    the points on the boundary of the object are colored red.

    Then the template is used to create a voxel grid. The voxel grid is a 3D array of
    -1s, 0s and 1s. The -1s represent the voxels outside the object, the 0s represent the
    voxels inside the object and the 1s represent the voxels on the boundary of the object.

    The method apply template will assign each point in the input point cloud value -1, 0 or 1
    based on the voxel grid. The points outside the object will be assigned -1, the points
    inside the object will be assigned 0 and the points on the boundary of the object will
    be assigned 1.

    """

    def __init__(self, num_rows: int, num_columns: int, bounding_box_size: tuple = (2, 1)):
        self.num_rows = num_rows
        self.num_columns = num_columns
        self.bounding_box_size = bounding_box_size
        self.template = np.zeros((num_rows, num_columns))

    def plot_template(self):
        """Plot the numbers in the template as the boxes of size 1x1. Plot also the boundaries between the boxes.
        """

        # Compute the width and height of each box in the template
        w, h = self.bounding_box_size / np.array([self.num_columns, self.num_rows])

        # Create a figure and axis
        fig, ax = plt.subplots()

        # Plot the template as the rectangles of size box_size x box_size
        for i in range(self.height):
            for j in range(self.width):
                if self.template[i, j] == 0:
                    color = 'b'
                elif self.template[i, j] == 1:
                    color = 'r'
                else:
                    color = 'g'

                # Plot rectangle of size 1
                rectangle = patches.Rectangle((j, i), 1, 1, linewidth=1, edgecolor='black', facecolor=color)
                ax.add_patch(rectangle)

        # Set axis limits
        x_range = (-1, self.width + 1)
        y_range = (-1, self.height + 1)
        ax.set_xlim(*x_range)
        ax.set_ylim(*y_range)

        # Set aspect ratio to equal for a square plot
        ax.set_aspect('equal', 'box')

        # Show the plot
        plt.show(block=False)

    def get_points(self):
        raise NotImplementedError


if __name__ == '__main__':
    rect_template = Template(6, 6)
    rect_template.plot_template()
    plt.show()
