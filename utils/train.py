import numpy as np
import progressbar
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.utils.data import ConcatDataset, DataLoader

from utils.dataset import (AmbientaDataset, PhysionetDataset, PostureClass,
                           SLPDataset)
from utils.model import ConvNet

################
#
# Hyper Parameters
#
################

num_trainings = 2
num_epochs = 2
learning_rate = 0.005
batch_size = 32
num_classes = len(PostureClass)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_data(transform):
    train_dataset = ConcatDataset(
        [
            PhysionetDataset(transform, train=True),
            SLPDataset(transform, train=True),
        ]
    )

    test_dataset = AmbientaDataset(transform, train=True)
    return train_dataset, test_dataset


def train(data):
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    model = ConvNet(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    n_total_steps = len(data_loader)
    prev_loss = 100

    widgets = [
        "Train progress: ",
        progressbar.Bar(left="[", right="]", marker="-"),
        " ",
        progressbar.Counter(format="%(value)d/%(max_value)d"),
        " ",
        progressbar.Variable('epoch', format='Epoch: {formatted_value}', width=3),
        " ",
        progressbar.Variable('loss', format='Loss: {formatted_value}'),
    ]

    loss_evolution = np.array([])
    accuracy_evolution = np.array([])
    with progressbar.ProgressBar(max_value=num_epochs*n_total_steps, widgets=widgets) as bar:
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(data_loader):
                labels = labels.long()
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(
                    images.unsqueeze(1)
                )  # Bring grayscale images from usual format (64x32) to a format with additional channel (1x64x32) (https://stackoverflow.com/questions/57237381/runtimeerror-expected-4-dimensional-input-for-4-dimensional-weight-32-3-3-but)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _, predictions = torch.max(outputs, 1)
                loss_evolution = np.append(loss_evolution, loss.item())
                accuracy_evolution = np.append(accuracy_evolution, (predictions == labels).sum().item() / labels.size(0))

                if i%100 == 0:
                    loss_char = f"\033[92m↘ {loss.item():.4f}\033[0m" if loss.item() < prev_loss else f"\033[91m↗ {loss.item():.4f}\033[0m"
                    bar.update(epoch * n_total_steps + i, loss=loss_char, epoch=f"{epoch+1}/{num_epochs}")
                    prev_loss = loss.item()

    return model, loss_evolution, accuracy_evolution


def evaluate(model, data, writer):
    data_loader = DataLoader(data, batch_size=batch_size)
    model.eval()
    predlist = []
    lbllist = []
    pr_labels = []
    pr_predictions = []
    acc = 0.0
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images.unsqueeze(1))

            _, predictions = torch.max(outputs, 1)

            lbllist.append(labels.cpu().numpy())
            predlist.append(predictions.cpu().numpy())
            class_predictions = [F.softmax(output, dim=0) for output in outputs]
            pr_predictions.append(class_predictions)
            pr_labels.append(predictions)
            n_samples += labels.size(0)
            n_correct += (predictions == labels).sum().item()

        acc = 100.0 * n_correct / n_samples
    pr_predictions = torch.cat([torch.stack(batch) for batch in pr_predictions])
    pr_labels = torch.cat(pr_labels)
    return confusion_matrix(np.concatenate(lbllist), np.concatenate(predlist)), pr_predictions, pr_labels
