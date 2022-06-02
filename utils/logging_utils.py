import datetime
from pathlib import Path
from typing import List
import torch
import torchvision
import numpy as np
from utils.dataset import PostureClass
from utils.train import HParams
from torch.utils.tensorboard import SummaryWriter

from utils.plots import plot_confusion_matrix


def write_samples_and_model(model, images, writer):
    torch.save(model.state_dict(), f"{writer.get_logdir()}/model.pt")
    image_grid = torchvision.utils.make_grid(images.unsqueeze(1))
    writer.add_image("samples", image_grid)
    writer.add_graph(model, images.unsqueeze(1))


def write_conf_mat(writer: SummaryWriter, conf_mat, title="Confusion matrix"):
    np.save(f"{writer.get_logdir()}/confmat.npy", conf_mat)
    fig = plot_confusion_matrix(conf_mat, normalize=True, title=title)
    writer.add_figure("confusion_matrix", fig)


def write_transform(writer: SummaryWriter, tag, transform: List):
    with open(f"{writer.get_logdir()}/transforms.txt", "a") as file:
        file.write(f"{tag}\n\t{str(transform)}\n")
    writer.add_text(f"transform/{tag}", " | ".join([str(t) for t in transform]))


def write_hyperparams(writer, hparams: HParams):
    with open(f"{writer.get_logdir()}/hyperparams.txt", "w") as file:
        file.write(
            "\n".join([f"{k}: {v}" for k, v in hparams.__dict__().items()] + [""])
        )
    writer.add_text(
        "hyperparameters",
        " | ".join([f"{k}: {v}" for k, v in hparams.__dict__().items()]),
    )


def write_scalars(writer, tag, values, sample_rate):
    for n_split, sub_values in enumerate(
        np.split(values, range(sample_rate, len(values), sample_rate))
    ):
        value = np.sum(sub_values) / len(sub_values)
        writer.add_scalar(tag, value, (sample_rate * (n_split - 1)) + len(sub_values))


def write_pr_curves(writer, pr_labels, pr_predictions):
    for run, (pr_lbls, pr_preds) in enumerate(zip(pr_labels, pr_predictions)):
        for i, label in enumerate(PostureClass):
            labels_i = pr_lbls == i
            preds_i = pr_preds[:, i]
            writer.add_pr_curve(str(label), labels_i, preds_i, global_step=run)
            writer.flush()


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
