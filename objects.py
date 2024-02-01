import numpy as np
import matplotlib.pyplot as plt

class Object(object):
    def __init__(self, obj_points: np.ndarray, out_points: np.ndarray):
        self.obj_points = obj_points
        self.out_points = out_points

    def transform(self, angle: float, translation: np.ndarray):
        raise NotImplementedError

    def visualize(self):
        raise NotImplementedError


class Rectangle(Object):
    def __init__(self, obj_points: np.ndarray, out_points: np.ndarray):
        super().__init__(obj_points, out_points)

    def visualize(self):
        # Plot the rectangle with the object and out points
        plt.plot(self.obj_points[:, 0], self.obj_points[:, 1], 'o', color='red')
        plt.plot(self.out_points[:, 0], self.out_points[:, 1], 'o', color='green')
        plt.show()

    def plot_template(self):
        plt.plot(self.obj_points[:, 0], self.obj_points[:, 1], 'o', color='red')
        plt.show()


