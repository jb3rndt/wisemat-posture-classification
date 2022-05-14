from typing import List
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib as mpl
import torchvision


def plot_samples(
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
    ax.imshow(image, cmap=cmap)
    ax.axis("off")
    if title:
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

def plot_samples_matrix(samples:List, row_length=10, cmap="gist_stern", figsize=(50, 50)):
    fig, rows = plt.subplots(len(samples) // row_length, row_length, figsize=figsize)
    for row_nr, row in enumerate(rows):
        for col_nr, col in enumerate(row):
            image, label = samples[row_nr * row_length + col_nr]
            plot_image(image, title=label, ax=col, cmap=cmap)
