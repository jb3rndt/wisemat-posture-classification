import datetime
from pathlib import Path

import numpy as np
import progressbar
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.dataset import PhysionetDataset, PostureClass, SLPDataset
from utils.model import ConvNet

################
#
# Hyper Parameters
#
################

num_trainings = 1
num_epochs = 1
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

    test_dataset = ConcatDataset(
        [
            PhysionetDataset(transform, train=False),
            SLPDataset(transform, train=False),
        ]
    )
    return train_dataset, test_dataset


def train(data, writer=None):
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    model = ConvNet(num_classes).to(device)
    images, labels = next(iter(data_loader))
    image_grid = torchvision.utils.make_grid(images.unsqueeze(1))
    writer.add_image("samples", image_grid)
    writer.add_graph(model, images.unsqueeze(1))

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

    running_loss = 0.0
    running_correct = 0
    with progressbar.ProgressBar(max_value=num_epochs*n_total_steps, widgets=widgets) as bar:
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(data_loader):
                labels = labels.long()
                # print(images.shape) # torch.Size([4, 1, 64, 32])
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(
                    images.unsqueeze(1)
                )  # Bring grayscale images from usual format (64x32) to a format with additional channel (1x64x32) (https://stackoverflow.com/questions/57237381/runtimeerror-expected-4-dimensional-input-for-4-dimensional-weight-32-3-3-but)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predictions = torch.max(outputs, 1)
                running_correct += (predictions == labels).sum().item()

                if (i+1)%10 == 0:
                    loss_char = f"\033[92m↘ {loss.item():.4f}\033[0m" if loss.item() < prev_loss else f"\033[91m↗ {loss.item():.4f}\033[0m"
                    bar.update(epoch * n_total_steps + i, loss=loss_char, epoch=f"{epoch+1}/{num_epochs}")
                    prev_loss = loss.item()
                    writer.add_scalar("train/loss", running_loss/10, epoch * n_total_steps + i)
                    writer.add_scalar("train/accuracy", running_correct/10, epoch * n_total_steps + i)
                    running_loss = 0.0
                    running_correct = 0
    writer.close()
    return model


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
    for i, label in enumerate(PostureClass):
        labels_i = pr_labels == i
        preds_i = pr_predictions[:, i]
        writer.add_pr_curve(str(label), labels_i, preds_i, global_step=0)
    writer.close()
    return confusion_matrix(np.concatenate(lbllist), np.concatenate(predlist))


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
        file.write(f"{str(transform.transforms)}\n")
    np.save(folder.joinpath("confmat.npy"), conf_mat)
