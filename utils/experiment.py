from contextlib import contextmanager
import datetime
import random
import numpy as np
import time
from typing import List
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from utils.dataset import PostureClass
from utils.logging_utils import (
    f1_scores_from_conf_mat,
    write_hyperparams,
    write_samples_and_model,
    write_scalars,
    write_conf_mat,
    write_transform,
    save_model,
)

from utils.plots import plot_confusion_matrix
from utils.train import (
    evaluate,
    read_data,
    train,
    num_epochs,
    batch_size,
    learning_rate,
    num_trainings,
)


class Experiment:
    def __init__(self, name: str, transform: List) -> None:
        self.name = name
        self.transform = torchvision.transforms.Compose(transform)

    def run(self):
        print(f"Running Experiment >>{self.name}<<")
        run_name = f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{self.name}'
        writer = SummaryWriter(
            f'runs/{run_name}'
        )
        with timed("Reading data"):
            train_dataset, test_dataset = read_data(self.transform)
            train_dataset = [
                train_dataset[index]
                for index in random.sample(range(len(train_dataset)), 100)
            ]
            test_dataset = [
                test_dataset[index]
                for index in random.sample(range(len(test_dataset)), 100)
            ]

        best_f1_score = 0.0
        best_model = None
        total_loss_evolution = []
        total_accuracy_evolution = []
        total_conf_mat = np.zeros((3, 3))
        for i in range(num_trainings):
            with timed(f"{i+1}. Training"):
                model, loss_evolution, accuracy_evolution = train(train_dataset)
                total_loss_evolution.append(loss_evolution)
                total_accuracy_evolution.append(accuracy_evolution)

            with timed(f"{i+1}. Evaluation"):
                conf_mat, pr_predictions, pr_labels = evaluate(model, test_dataset, writer)
                sum_f1_score = sum(f1_scores_from_conf_mat(conf_mat))
                if sum_f1_score > best_f1_score:
                    best_f1_score = sum_f1_score
                    best_model = model
                total_conf_mat += conf_mat
                for i, label in enumerate(PostureClass):
                    labels_i = pr_labels == i
                    preds_i = pr_predictions[:, i]
                    writer.add_pr_curve(str(label), labels_i, preds_i, global_step=i)
                    writer.flush()

        images = list(map(lambda s: s[0], random.sample(train_dataset, 32)))
        write_samples_and_model(best_model, torch.stack(images), writer)

        total_loss_evolution = (
            np.sum(np.stack(total_loss_evolution), axis=0) / num_trainings
        )
        total_accuracy_evolution = (
            np.sum(np.stack(total_accuracy_evolution), axis=0) / num_trainings
        )
        write_scalars(writer, "train/loss", total_loss_evolution, 100)
        write_scalars(writer, "train/accuracy", total_accuracy_evolution, 100)

        total_conf_mat /= num_trainings
        plot_confusion_matrix(
            total_conf_mat, normalize=True, title=f"Confusion Matrix of {self.name}"
        )
        write_conf_mat(writer, total_conf_mat, title=f"Confusion Matrix of {self.name}")
        write_transform(writer, self.transform.transforms)
        write_hyperparams(
            writer,
            {
                "num_trainings": num_trainings,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
            },
        )
        save_model(best_model, self.transform, total_conf_mat)
        writer.flush()
        writer.close()
        print(f"\033[92mSuccessfully ran Experiment >>{self.name}<<\033[0m")


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
