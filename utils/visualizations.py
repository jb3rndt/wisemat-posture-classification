from typing import List
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib as mpl
import torchvision
import utils.dataset as ds
import cv2
import numpy as np


def _plot_samples(
    dataset,
    sample_indices,
    classes=None,
    transform: torchvision.transforms.Compose = torchvision.transforms.Compose([]),
):
    assert classes is None or len(sample_indices) == len(
        classes
    ), "Number of samples must be equal to number of classes"
    fig, axs = plt.subplots(1, len(sample_indices), figsize=(20, 4))
    for col_nr, col in enumerate(axs):
        if not sample_indices[col_nr]:
            continue
        image, label = transform(dataset[sample_indices[col_nr]])
        plot_image(image, ax=col, title=classes[label])

    cax = plt.axes([0.1, 0.1, 0.8, 0.075])
    fig.colorbar(
        cm.ScalarMappable(
            norm=mpl.colors.Normalize(vmin=-1, vmax=1), cmap="gist_stern"
        ),
        cax=cax,
        orientation="horizontal",
        pad=0.2,
    )


def plot_image(image, title=None, cmap="gist_stern", ax=None):
    image = reshape_image(image)
    if not ax:
        _, ax = plt.subplots()
    # ax.hist(image.ravel(), bins=256)
    ax.imshow(image, cmap=cmap)
    ax.axis("off")
    if title is not None:
        ax.set_title(title)
    return ax


def reshape_image(image):
    if len(image.shape) == 2:
        return image.reshape(image.shape + tuple([1]))
    color_channels = min(image.shape)
    shape_list = list(image.shape)
    shape_list.remove(color_channels)
    new_shape = tuple(shape_list) + (color_channels,)
    return image.reshape(new_shape)


def image_row(*images: List, cmap="gist_stern", figsize=(10, 10)):
    fig, axs = plt.subplots(1, len(images), figsize=figsize)
    for col, image in zip(axs, images):
        plot_image(image, ax=col, cmap=cmap)


def plot_samples(
    samples: List,
    ncols=None,
    cmap="gist_stern",
    figsize=None,
    label_convert=lambda l: ds.PostureClass(l),
    transform: torchvision.transforms.Compose = torchvision.transforms.Compose([]),
):
    nrows = len(samples) // ncols
    if ncols is None:
        ncols = len(samples)
    if figsize is None:
        figsize = (ncols * 5, nrows * 10)
    fig, rows = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1:
        rows = [rows]
    for row_nr, row in enumerate(rows):
        for col_nr, col in enumerate(row):
            image, label = transform(samples[row_nr * ncols + col_nr])
            plot_image(image, title=label_convert(label), ax=col, cmap=cmap)


def apply_lines(image, lines, n=1):
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    lines = lines[:n]
    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(image, (x1, y1), (x2, y2), (255, 145, 0), 1)
    return image
