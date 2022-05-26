import datetime
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from utils.dataset import AmbientaDataset, PhysionetDataset, classes
from utils.model import ConvNet
from utils.plots import (plot_class_weights, plot_comparing_confusion_matrix,
                         plot_confusion_matrix)
from utils.transforms import (Blur, Close, EqualizeHist, Erode, Normalize,
                              Resize, Threshold, ToTensor)

################
#
# Hyper Parameters
#
################

num_trainings = 1
num_epochs = 1
learning_rate = 0.005
batch_size = 1000
num_classes = len(classes)

composed_transforms = torchvision.transforms.Compose(
    [
        Resize((26, 64)),
        Normalize(),
        EqualizeHist(),
        Blur((5, 5)),
        Erode(),
        # Threshold(),
        Resize((52, 128)),
        ToTensor(),
    ]
)

def main():

    ################
    #
    # Data Reading & Preprocessing
    #
    ################

    train_dataset = ConcatDataset(
        [
            PhysionetDataset(composed_transforms, train=True),
            AmbientaDataset(composed_transforms, train=True),
        ]
    )

    test_dataset = ConcatDataset(
        [
            PhysionetDataset(composed_transforms, train=False),
            AmbientaDataset(composed_transforms, train=False),
        ]
    )

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of testing samples: {len(test_dataset)}")

    # Over- & Undersampling
    train_labels = np.concatenate(
        list(map(lambda dataset: dataset.y, train_dataset.datasets))
    )
    test_labels = np.concatenate(
        list(map(lambda dataset: dataset.y, test_dataset.datasets))
    )
    _, train_class_counts = np.unique(train_labels, return_counts=True)
    _, test_class_counts = np.unique(test_labels, return_counts=True)
    weights = np.asarray([1.0 / train_class_counts[c] for c in train_labels])
    train_sampler = WeightedRandomSampler(
        weights=weights, num_samples=len(weights), replacement=True
    )

    conf_mat_sum = np.zeros((11, 11))
    conf_mats = []
    finished = 2
    for i in range(num_trainings):
        conf_mat, acc = train_model(train_dataset, test_dataset, train_sampler, save_model=True)
        print(f"Accuracy of {i+1}. Network: {acc:.4f}")
        print(conf_mat)
        conf_mats.append(conf_mat)
        conf_mat_sum += conf_mat
        finished += 1

        with open(f'benchmarks/test.npy', 'wb') as f:
            np.save(f, conf_mat)

    # f1_scores = f1_scores_from_conf_mat(conf_mat_sum)
    # mean_score = sum(f1_scores) / len(f1_scores)

    # print(conf_mat_sum)


    plot_confusion_matrix(conf_mat_sum, classes, normalize=True)
    plt.show()


def train_model(train_dataset, test_dataset, train_sampler, save_model=False):
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    ################
    #
    # Machine Learning Part
    #
    ################

    # device config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConvNet(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            labels = labels.long()
            # print(images.shape) # torch.Size([4, 1, 64, 32])
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            # print(outputs.shape, labels.shape) # torch.Size([4, 5]) torch.Size([4])
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 50 == 0:
                print(
                    f"Epoch {epoch+1} / {num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}"
                )

    ################
    #
    # Evaluation Part
    #
    ################

    # Test the model
    model.eval()
    predlist = []
    lbllist = []
    acc = 0.0
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            _, predictions = torch.max(outputs, 1)

            lbllist.append(labels.cpu().numpy())
            predlist.append(predictions.cpu().numpy())
            n_samples += labels.size(0)
            n_correct += (predictions == labels).sum().item()

        acc = 100.0 * n_correct / n_samples

        # print(f1_score(np.concatenate(lbllist), np.concatenate(predlist), average='macro'))
        # print(f1_score(np.concatenate(lbllist), np.concatenate(predlist), average=None))

    # Save model if specified
    if save_model:
        folder: Path = Path("models").joinpath(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        Path.mkdir(folder, parents=True, exist_ok=True)
        torch.save(model.state_dict(), f"{folder}/model.pt")
        with open(f"{folder}/hyperparams.txt", "w") as file:
            file.write(f"learning_rate = {learning_rate}\nnum_epochs = {num_epochs}\nbatch_size = {batch_size}\n")
            file.write(str(composed_transforms.transforms))

    return confusion_matrix(np.concatenate(lbllist), np.concatenate(predlist)), acc


def f1_scores_from_conf_mat(cm):
    f1_scores = []
    for i in range(cm.shape[0]):
        precision = cm[i, i] / sum(cm[:, i]) if sum(cm[:, i]) else 0
        recall = cm[i, i] / sum(cm[i, :]) if sum(cm[i, :]) else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
        f1_scores.append(f1_score)

    return f1_scores


if __name__ == "__main__":
    main()
