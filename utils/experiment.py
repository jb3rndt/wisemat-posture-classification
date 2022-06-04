from contextlib import contextmanager
import datetime
import random
import numpy as np
import time
from typing import List, Optional
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from utils.dataset import PostureClass
from utils.logging_utils import (
    f1_scores_from_conf_mat,
    write_hyperparams,
    write_pr_curves,
    write_samples_and_model,
    write_scalars,
    write_conf_mat,
    write_transform,
)
from utils.model import ConvNet

from utils.plots import plot_confusion_matrix
from utils.train import evaluate, read_data, train, HParams


class Experiment:
    def __init__(
        self,
        name: str,
        transform: List,
        slp_transform: Optional[List] = None,
        hparams=HParams(),
    ) -> None:
        self.name = name
        self.hparams = hparams
        self.physionet_transform = torchvision.transforms.Compose(transform)
        self.slp_transform = torchvision.transforms.Compose(slp_transform or transform)

    @classmethod
    def reevaluate(cls, full_name: str, transform: List) -> "Experiment":
        state_dict = torch.load(f"runs/{full_name}/model.pt")
        model = ConvNet(len(PostureClass))
        model.load_state_dict(state_dict)
        __, test_dataset = read_data(
            torchvision.transforms.Compose(transform),
            torchvision.transforms.Compose(transform),
        )
        conf_mat, __, __ = evaluate(model, test_dataset, HParams())
        plot_confusion_matrix(conf_mat, normalize=True, title=f"Confusion Matrix")

    def run(self):
        print(f"Running Experiment >>{self.name}<<")
        run_name = (
            f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{self.name}'
        )

        with timed("Reading data"):
            train_dataset, test_dataset = read_data(
                self.physionet_transform, self.slp_transform
            )
            # train_dataset = [
            #     train_dataset[index]
            #     for index in random.sample(range(len(train_dataset)), 100)
            # ]
            # test_dataset = [
            #     test_dataset[index]
            #     for index in random.sample(range(len(test_dataset)), 100)
            # ]

        best_f1_score = 0.0
        best_model = None
        total_loss_evolution = []
        total_accuracy_evolution = []
        total_pr_predictions = []
        total_pr_labels = []
        total_conf_mat = np.zeros((3, 3))
        for i in range(self.hparams.num_trainings):
            with timed(f"{i+1}. Training"):
                model, loss_evolution, accuracy_evolution = train(
                    train_dataset, self.hparams
                )
                total_loss_evolution.append(loss_evolution)
                total_accuracy_evolution.append(accuracy_evolution)

            with timed(f"{i+1}. Evaluation"):
                conf_mat, pr_predictions, pr_labels = evaluate(
                    model, test_dataset, self.hparams
                )
                sum_f1_score = sum(f1_scores_from_conf_mat(conf_mat))
                if sum_f1_score > best_f1_score:
                    best_f1_score = sum_f1_score
                    best_model = model
                total_conf_mat += conf_mat
                total_pr_labels.append(pr_labels)
                total_pr_predictions.append(pr_predictions)

        images = list(
            map(
                lambda i: train_dataset[i][0],
                random.sample(range(len(train_dataset)), 32),
            )
        )

        total_loss_evolution = (
            np.sum(np.stack(total_loss_evolution), axis=0) / self.hparams.num_trainings
        )
        total_accuracy_evolution = (
            np.sum(np.stack(total_accuracy_evolution), axis=0)
            / self.hparams.num_trainings
        )

        total_conf_mat /= self.hparams.num_trainings
        plot_confusion_matrix(
            total_conf_mat, normalize=True, title=f"Confusion Matrix of {self.name}"
        )

        self.save(
            run_name,
            best_model,
            total_conf_mat,
            total_loss_evolution,
            total_accuracy_evolution,
            images,
            total_pr_labels,
            total_pr_predictions,
        )
        print(f"\033[92mSuccessfully ran Experiment >>{self.name}<<\033[0m")

    def save(
        self,
        run_name: str,
        model: torch.nn.Module,
        conf_mat: np.ndarray,
        loss_evolution,
        accuracy_evolution,
        samples,
        pr_labels,
        pr_predictions,
    ):
        writer = SummaryWriter(f"runs/{run_name}")
        write_samples_and_model(model, torch.stack(samples), writer)
        write_scalars(writer, "train/loss", loss_evolution, 100)
        write_scalars(writer, "train/accuracy", accuracy_evolution, 100)
        write_conf_mat(writer, conf_mat, title=f"Confusion Matrix of {self.name}")
        write_transform(writer, "physionet", self.physionet_transform.transforms)
        write_transform(writer, "slp", self.slp_transform.transforms)
        write_hyperparams(writer, self.hparams)
        write_pr_curves(writer, pr_labels, pr_predictions)
        writer.flush()
        writer.close()


@contextmanager
def timed(name):
    def passed_time(start_time):
        diff = time.time() - start_time
        time_str = ""
        hours = int(diff // 3600)
        if hours > 0:
            time_str += f"{hours}h "
        minutes = int(diff // 60 % 60)
        if minutes > 0:
            time_str += f"{minutes}m "
        seconds = diff % 60
        time_str += f"{seconds:.1f}s"
        return time_str

    start_time = time.time()
    try:
        yield
        print(f"\033[92m{name} took {passed_time(start_time)}\033[0m")
    except Exception as e:
        print(f"\033[91m{name} failed after {passed_time(start_time)}\033[0m")
        raise e from None
