import itertools
from math import isqrt
from typing import List

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import utils.dataset as ds
from utils.dataset import PostureClass


# from https://deeplizard.com/learn/video/0LhiS6yu2qQ
def plot_confusion_matrix(
    cm, normalize=False, title="Confusion matrix", cmap=plt.cm.Oranges
):
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    # plt.rcParams.update({"font.size": 22})
    classes = [str(c) for c in PostureClass]
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca()
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title, fontsize=22)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=22)
    plt.yticks(tick_marks, classes, fontsize=22)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=22,
        )

    plt.ylabel("Wahrheit", fontsize=22)
    # im.axes.tick_params(color="#CACACA", labelcolor="#CACACA")
    # ax.xaxis.label.set_color("#CACACA")
    # ax.yaxis.label.set_color("#CACACA")
    plt.xlabel("Vorhersage", fontsize=22)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.25)
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.ax.tick_params(labelsize=22)
    # cb.outline.set_edgecolor("#CACACA")
    # plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="#CACACA")

    # for spine in im.axes.spines.values():
    #     spine.set_edgecolor("#CACACA")

    fig.tight_layout()
    return fig


def plot_comparing_confusion_matrix(
    base_cm, compare_cm, classes, normalize=False, title="Confusion matrix"
):
    if normalize:
        base_cm = base_cm.astype("float") / base_cm.sum(axis=1)[:, np.newaxis]
        compare_cm = compare_cm.astype("float") / compare_cm.sum(axis=1)[:, np.newaxis]

    cm = compare_cm - base_cm
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if i != j:
            cm[i, j] *= -1

    plt.figure(figsize=(10, 10))
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "rg", ["r", "r", "w", "g", "g"], N=256
    )
    ax = plt.gca()
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap, vmin=-1, vmax=1)
    # plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        n = cm[i, j] if i == j else cm[i, j] * -1
        number = float(int(n)) if int(n * 100) == 0 else n
        plt.text(
            j,
            i,
            f"{'+' if number > 0 else ''}{format(number, fmt)}",
            horizontalalignment="center",
            color="black",
            weight="bold" if i == j else "normal",
        )

    plt.ylabel("True label")
    im.axes.tick_params(color="#CACACA", labelcolor="#CACACA")
    ax.xaxis.label.set_color("#CACACA")
    ax.yaxis.label.set_color("#CACACA")
    plt.xlabel("Predicted label")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.25)
    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.ax.yaxis.set_tick_params(color="#CACACA")
    cb.outline.set_edgecolor("#CACACA")
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="#CACACA")

    for spine in im.axes.spines.values():
        spine.set_edgecolor("#CACACA")

    plt.tight_layout()


def plot_class_weights(weights_list, classes, legend_labels=[]):
    legend_labels += [None] * len(weights_list)
    bar_width = 0.8 / len(weights_list)
    x = np.arange(len(classes))
    for i, (weights, label) in enumerate(zip(weights_list, legend_labels)):
        bar = plt.bar(x + i * bar_width, weights, bar_width)
        if label:
            bar.set_label(label)

    plt.title("Relative Classes distribution")

    print(x + bar_width / 2)
    plt.xticks(x + bar_width / 2, classes, rotation=45)
    plt.ylabel("Class Weight")
    plt.tight_layout()

    if legend_labels[0]:
        plt.legend()


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
    plt.rcParams.update({"font.size": 20})
    image = reshape_image(image)
    if not ax:
        _, ax = plt.subplots(figsize=(10, 10))
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


def image_row(*images: List, titles=[], cmap="gist_stern", figsize=(10, 10), gridspec_kw=None):
    titles += [None] * (len(images) - len(titles))
    fig, axs = plt.subplots(1, len(images), figsize=figsize, gridspec_kw=gridspec_kw)
    for i, (col, image) in enumerate(zip(axs, images)):
        plot_image(image, title=titles[i], ax=col, cmap=cmap)


def plot_samples(
    samples: List,
    ncols=None,
    cmap="gist_stern",
    figsize=None,
    label_convert=lambda l: ds.PostureClass(l),
    transform: torchvision.transforms.Compose = torchvision.transforms.Compose([]),
):
    if ncols is None:
        ncols = isqrt(len(samples)) + 1
    nrows = len(samples) // ncols
    if figsize is None:
        figsize = (ncols * 5, nrows * 10)
    fig, rows = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1:
        rows = [rows]
    for row_nr, row in enumerate(rows):
        for col_nr, col in enumerate(row):
            image, label = samples[row_nr * ncols + col_nr]
            plot_image(transform(image), title=label_convert(label), ax=col, cmap=cmap)


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


def plot_wrong_predictions(images, labels, predictions):
    wrong_predictions = []
    for i, (label, prediction) in enumerate(zip(labels, predictions)):
        if label != prediction:
            wrong_predictions.append(
                (images[i].squeeze(), (label.item(), prediction.item()))
            )
    plot_samples(
        wrong_predictions,
        label_convert=lambda x: f"Label: {ds.PostureClass(x[0])}\nPrediction: {ds.PostureClass(x[1])}",
    )
