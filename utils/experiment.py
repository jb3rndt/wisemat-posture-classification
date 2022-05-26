import datetime
import random
from typing import List
import torchvision
from torch.utils.tensorboard import SummaryWriter

from utils.plots import plot_confusion_matrix
from utils.train import evaluate, read_data, save_model, train


class Experiment:
    def __init__(self, name: str, transform: List) -> None:
        self.name = name
        self.transform = torchvision.transforms.Compose(transform)

    def run(self):
        print(f"Running Experiment >>{self.name}<<")
        writer = SummaryWriter(f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{self.name}')
        train_dataset, test_dataset = read_data(self.transform)
        # train_dataset = [train_dataset[index] for index in random.sample(range(len(train_dataset)), 100)]
        # test_dataset = [test_dataset[index] for index in random.sample(range(len(test_dataset)), 100)]
        model = train(train_dataset, writer)
        conf_mat = evaluate(model, test_dataset)
        save_model(model, self.transform, conf_mat)
        plot_confusion_matrix(conf_mat, normalize=True)
        writer.close()
        print(f"\033[92mSuccessfully ran Experiment >>{self.name}<<\033[0m")
