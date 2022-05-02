from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib as mpl
import torchvision


def plot_samples(
    dataset,
    sample_indices,
    classes,
    transform: torchvision.transforms.Compose = torchvision.transforms.Compose([]),
):
    assert len(sample_indices) == len(
        classes
    ), "Number of samples must be equal to number of classes"
    fig, axs = plt.subplots(1, len(classes), figsize=(20, 4))
    for col_nr, col in enumerate(axs):
        if not sample_indices[col_nr]:
            col.set_xticks([], [])
            col.set_yticks([], [])
            continue
        image, label = transform(dataset[sample_indices[col_nr]])
        col.imshow(image[0], origin="lower", cmap="gist_stern")
        col.set_title(classes[label])
        col.set_xticks([], [])
        col.set_yticks([], [])

    cax = plt.axes([0.1, 0.1, 0.8, 0.075])
    fig.colorbar(cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=-1, vmax=1), cmap="gist_stern"), cax=cax, orientation='horizontal', pad=0.2)

def plot_image(image, title=None, cmap=None, ax=None):
    if not ax:
        _, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image[0], origin="lower", cmap=cmap)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)
    return ax
