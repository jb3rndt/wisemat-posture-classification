import datetime
from pathlib import Path
from typing import List
import torch
import torchvision
import numpy as np

from utils.plots import plot_confusion_matrix
from utils.train import (
    num_epochs,
    batch_size,
    learning_rate,
    num_trainings,
)


def write_samples_and_model(model, images, writer):
    image_grid = torchvision.utils.make_grid(images.unsqueeze(1))
    writer.add_image("samples", image_grid)
    writer.add_graph(model, images.unsqueeze(1))


def write_conf_mat(writer, conf_mat, title="Confusion matrix"):
    fig = plot_confusion_matrix(conf_mat, normalize=True, title=title)
    writer.add_figure("confusion_matrix", fig)


def write_transform(writer, transform: List):
    writer.add_text("transform", " | ".join([str(t) for t in transform]))


def write_hyperparams(writer, hyperparams):
    writer.add_text(
        "hyperparameters", " | ".join([f"{k}: {v}" for k, v in hyperparams.items()])
    )


def write_scalars(writer, tag, values, sample_rate):
    for n_split, sub_values in enumerate(
        np.split(values, range(sample_rate, len(values), sample_rate))
    ):
        value = np.sum(sub_values) / len(sub_values)
        writer.add_scalar(tag, value, (sample_rate * (n_split - 1)) + len(sub_values))


def save_model(model, transform, conf_mat):
    folder: Path = (
        Path("models")
        .joinpath("autosave")
        .joinpath(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    )
    Path.mkdir(folder, parents=True, exist_ok=True)
    torch.save(model.state_dict(), folder.joinpath("model.pt"))
    with open(folder.joinpath("hyperparams.txt"), "w") as file:
        file.write(f"learning_rate = {learning_rate}\n")
        file.write(f"num_epochs = {num_epochs}\n")
        file.write(f"batch_size = {batch_size}\n")
        file.write(f"num_trainings = {num_trainings}\n")
        file.write(f"{str(transform.transforms)}\n")
    np.save(folder.joinpath("confmat.npy"), conf_mat)


def f1_scores_from_conf_mat(cm):
    f1_scores = []
    for i in range(cm.shape[0]):
        precision = cm[i, i] / sum(cm[:, i]) if sum(cm[:, i]) else 0
        recall = cm[i, i] / sum(cm[i, :]) if sum(cm[i, :]) else 0
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall)
            else 0
        )
        f1_scores.append(f1_score)

    return f1_scores
