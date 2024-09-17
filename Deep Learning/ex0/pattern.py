import numpy as np
import matplotlib.pyplot as plt

class Checker:

    def __init__(self, resolution, tile_size):
        if resolution % (2 * tile_size) != 0:
            raise ValueError

        self.resolution = resolution
        self.tile_size = tile_size
        self.output = None

    def draw(self):
        black = np.zeros((self.tile_size, self.tile_size))
        white = np.ones((self.tile_size, self.tile_size))
        row1 = np.concatenate((black, white), axis = 1)
        row2 = np.concatenate((white, black), axis = 1)
        pattern = np.concatenate((row1, row2))
        number = self.resolution // (2 * self.tile_size)
        self.output = np.tile(pattern,(number, number))
        return self.output.copy()

    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.show()

class Circle:

    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.x0, self.y0 = position
        self.output = np.zeros((self.resolution, self.resolution))

    def draw(self):
        x, y = np.meshgrid(range(self.resolution), range(self.resolution))
        circle = (x - self.x0)**2 + (y - self.y0)**2 < self.radius**2
        self.output[circle] = 1
        return self.output.copy()

    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.show()

class Spectrum:

    def __init__(self, resolution):
        self.resolution = resolution
        self.output = np.zeros((self.resolution, self.resolution, 3))

    def draw(self):
        intensity = np.linspace(0, 1, self.resolution)
        direction = np.outer(np.ones(self.resolution), intensity)
        self.output[:, :, 0] = direction #red
        self.output[:, :, 1] = direction.T #green
        rearranged = intensity[::-1]
        self.output[:, :, 2] = np.outer(np.ones(self.resolution), rearranged) #blue
        return self.output.copy()

    def show(self):
        plt.imshow(self.output)
        plt.show()
