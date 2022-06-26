from matplotlib import pyplot as plt
import numpy as np
import progressbar
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.utils.data import ConcatDataset, DataLoader

from utils.dataset import AmbientaDataset, PhysionetDataset, PostureClass, SLPDataset
from utils.model import ConvNet
from utils.plots import plot_samples, plot_wrong_predictions

################
#
# Hyper Parameters
#
################


class HParams:
    def __init__(
        self,
        learning_rate: float = 0.005,
        num_epochs: int = 10,
        batch_size: int = 32,
        num_trainings: int = 5,
    ):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_trainings = num_trainings

    def __dict__(self):
        return {
            "num_trainings": self.num_trainings,
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
        }


num_classes = len(PostureClass)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_data(transform_physionet, transform_slp):
    train_dataset = ConcatDataset(
        [
            PhysionetDataset(transform_physionet, train=True),
            SLPDataset(transform_slp, train=True),
        ]
    )

    test_dataset = AmbientaDataset(transform_physionet, train=True)
    return train_dataset, test_dataset


def train(model: ConvNet, data, hparams: HParams):
    data_loader = DataLoader(data, batch_size=hparams.batch_size, shuffle=True)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=hparams.learning_rate)

    n_total_steps = len(data_loader)
    prev_loss = 100

    widgets = [
        "Train progress: ",
        progressbar.Bar(left="[", right="]", marker="-"),
        " ",
        progressbar.Counter(format="%(value)d/%(max_value)d"),
        " ",
        progressbar.Variable("epoch", format="Epoch: {formatted_value}", width=3),
        " ",
        progressbar.Variable("loss", format="Loss: {formatted_value}"),
    ]

    loss_evolution = np.array([])
    accuracy_evolution = np.array([])
    with progressbar.ProgressBar(
        max_value=hparams.num_epochs * n_total_steps, widgets=widgets
    ) as bar:
        for epoch in range(hparams.num_epochs):
            for i, (images, labels) in enumerate(data_loader):
                images = images.float()
                labels = labels.long()
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _, predictions = torch.max(outputs, 1)
                loss_evolution = np.append(loss_evolution, loss.item())
                accuracy_evolution = np.append(
                    accuracy_evolution,
                    (predictions == labels).sum().item() / labels.size(0),
                )

                if (i + 1) % 10 == 0:
                    loss_char = (
                        f"\033[92m↘ {loss.item():.4f}\033[0m"
                        if loss.item() < prev_loss
                        else f"\033[91m↗ {loss.item():.4f}\033[0m"
                    )
                    bar.update(
                        epoch * n_total_steps + i,
                        loss=loss_char,
                        epoch=f"{epoch+1}/{hparams.num_epochs}",
                    )
                    prev_loss = loss.item()

    return loss_evolution, accuracy_evolution


def evaluate(model, data, hparams: HParams):
    data_loader = DataLoader(data, batch_size=hparams.batch_size, shuffle=True)
    model.eval()
    predlist = []
    lbllist = []
    imglist = []
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            imglist.append(images.cpu())
            lbllist.append(labels.cpu())
            predlist.append(
                torch.stack([F.softmax(output, dim=0) for output in outputs]).cpu()
            )
    return (torch.cat(imglist), torch.cat(lbllist), torch.cat(predlist))
