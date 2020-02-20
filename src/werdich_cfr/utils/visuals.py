""" Some functions to help with visualizations """
# imports
import numpy as np
import ipywidgets as ipyw
from matplotlib import pyplot as plt


class ImageSliceViewer3D:
    """
    https://github.com/mohakpatel/ImageSliceViewer3D/blob/master/ImageSliceViewer3D.ipynb
    """

    def __init__(self, volume, figsize=(8, 8), cmap='gray'):
        self.volume = volume
        self.figsize = figsize
        self.cmap = cmap
        self.v = [np.min(volume), np.max(volume)]

        # Call to select slice plane
        ipyw.interact(self.view_selection, view=ipyw.RadioButtons(
            options=['x-y', 'y-z', 'z-x'], value='x-y',
            description='Slice plane selection:', disabled=False,
            style={'description_width': 'initial'}))

    def view_selection(self, view):
        # Transpose the volume to orient according to the slice plane selection
        orient = {"y-z": [1, 2, 0], "z-x": [2, 0, 1], "x-y": [0, 1, 2]}
        self.vol = np.transpose(self.volume, orient[view])
        maxZ = self.vol.shape[2] - 1

        # Call to view a slice within the selected slice plane
        ipyw.interact(self.plot_slice,
                      z=ipyw.IntSlider(min=0, max=maxZ, step=1, continuous_update=False,
                                       description='Image Slice:'))

    def plot_slice(self, z):
        # Plot slice for the given plane and slice
        self.fig = plt.figure(figsize=self.figsize)
        plt.imshow(self.vol[:, :, z], cmap=plt.get_cmap(self.cmap),
                   vmin=self.v[0], vmax=self.v[1])
