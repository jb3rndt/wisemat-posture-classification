import matplotlib.pyplot as plt
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
