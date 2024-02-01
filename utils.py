import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import matplotlib
matplotlib.use('TkAgg')

def plot_rectangle(width, height, color='b'):

    # Create a figure and axis
    fig, ax = plt.subplots()

    center = (-width / 2, -height / 2)

    # Plot rectangle of size 1
    rectangle1 = patches.Rectangle(center, width, height, linewidth=1, edgecolor=color, facecolor='none')
    ax.add_patch(rectangle1)

    # Plot the point (0, 0)
    plt.plot(0, 0, 'o', color='black')

    # Set axis limits
    x_range = (- (width + 1), width + 1)
    y_range = (- (height + 1), height + 1)
    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)

    # Set aspect ratio to equal for a square plot
    ax.set_aspect('equal', 'box')

    # Show the plot
    plt.show(block=False)

def plot_ellipse(width, height, color='b'):
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot ellipse
    ellipse = patches.Ellipse((0, 0), width, height, linewidth=1, edgecolor=color, facecolor='none')
    ax.add_patch(ellipse)

    # Plot the point (0, 0)
    plt.plot(0, 0, 'o', color='black')

    # Set axis limits
    x_range = (- (width + 1), width + 1)
    y_range = (- (height + 1), height + 1)

    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)

    # Set aspect ratio to equal for a square plot
    ax.set_aspect('equal', 'box')

    # Show the plot
    plt.show(block=False)


def get_points():
    """Get the points from the user interactively using the ginput function."""
    # Get the points from the user
    points = plt.ginput(n=-1, timeout=-1, show_clicks=True)
    points = np.array(points)
    return points


def get_rectangle_object():

    # Plot the rectangle
    plot_rectangle(2, 1)

    plt.title('Select the object points')
    obj_points = get_points()

    # Plot the rectangle with the object points
    plot_rectangle(2, 1)
    plt.plot(obj_points[:, 0], obj_points[:, 1], 'o', color='red')

    plt.title('Select the out points')
    out_points = get_points()

    # Plot the rectangle with the object and out points
    plot_rectangle(2, 1)
    plt.plot(obj_points[:, 0], obj_points[:, 1], 'o', color='red')
    plt.plot(out_points[:, 0], out_points[:, 1], 'o', color='green')
    plt.show()

    return obj_points, out_points

def get_ellipse_object():

    # Plot the ellipse
    plot_ellipse(2, 1)

    plt.title('Select the object points')
    obj_points = get_points()

    # Plot the ellipse with the object points
    plot_ellipse(2, 1)
    plt.plot(obj_points[:, 0], obj_points[:, 1], 'o', color='red')

    plt.title('Select the out points')
    out_points = get_points()

    # Plot the ellipse with the object and out points
    plot_ellipse(2, 1)
    plt.plot(obj_points[:, 0], obj_points[:, 1], 'o', color='red')
    plt.plot(out_points[:, 0], out_points[:, 1], 'o', color='green')
    plt.show()

    return obj_points, out_points

def save_object(obj_points: np.ndarray, out_points: np.ndarray, file_name: str):
    """Save the object and out points to a file."""
    # Save the points to a file
    file = os.path.join('data', file_name)
    np.savez(file, obj_points=obj_points, out_points=out_points)

def visualize_object(obj_points: np.ndarray, out_points: np.ndarray, obj_type: str):
    plt.plot(obj_points[:, 0], obj_points[:, 1], 'o', color='red')
    plt.plot(out_points[:, 0], out_points[:, 1], 'o', color='green')

def load_object(file_name: str):
    """Load the object and out points from a file."""
    # Load the points from a file
    file = os.path.join('data', file_name)
    data = np.load(file)
    obj_points = data['obj_points']
    out_points = data['out_points']
    return obj_points, out_points

def transform_object(obj_points: np.ndarray, out_points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Transform the object and out points using the transformation matrix."""
    # Transform the points
    obj_points_transformed = np.dot(R, obj_points.T).T + t.T
    out_points_transformed = np.dot(R, out_points.T).T + t.T
    return obj_points_transformed, out_points_transformed

def generate_transformation():
    """Generate a random transformation matrix."""
    # Generate a random transformation matrix

    angle = np.random.rand() * 2 * np.pi
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    t = np.random.rand(2, 1)

    print(f"Generated rotation angle: {angle}")
    print(f"Generated translation: {t}")
    return R, t

if __name__ == '__main__':

    # obj_points, out_points = get_rectangle_object()
    # save_object(obj_points, out_points, 'rectangle.npz')
    obj_points, out_points = load_object('rectangle.npz')
    plot_rectangle(2, 1)
    visualize_object(obj_points, out_points, 'rectangle')
    plt.show()

    R, t = generate_transformation()
    obj_points_transformed, out_points_transformed = transform_object(obj_points, out_points, R, t)
    plot_rectangle(2, 1)
    visualize_object(obj_points_transformed, out_points_transformed, 'rectangle')
    plt.quiver(0, 0, t[0], t[1], color='black', angles='xy', scale_units='xy', scale=1)
    plt.show()

    # get_ellipse_object()
