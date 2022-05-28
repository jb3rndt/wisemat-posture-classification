from contextlib import contextmanager
import datetime
import random
import time
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
        writer = SummaryWriter(
            f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{self.name}'
        )
        with timed("Reading data"):
            train_dataset, test_dataset = read_data(self.transform)
            # train_dataset = [train_dataset[index] for index in random.sample(range(len(train_dataset)), 100)]
            # test_dataset = [test_dataset[index] for index in random.sample(range(len(test_dataset)), 100)]
        with timed("Training"):
            model = train(train_dataset, writer)
        with timed("Evaluating"):
            conf_mat = evaluate(model, test_dataset, writer)
        save_model(model, self.transform, conf_mat)
        plot_confusion_matrix(conf_mat, normalize=True)
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
