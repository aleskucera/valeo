import torch
import pytorch3d
import numpy as np
import matplotlib.pyplot as plt
from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import plot_batch_individually


class Template(object):
    def __init__(self, shape: tuple):
        self.shape = shape
        self.points = Pointclouds(points=[torch.zeros(shape)])

    def visualize(self):
        plot_batch_individually(self.points)
