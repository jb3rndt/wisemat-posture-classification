from pathlib import Path
import random
import numpy as np
import pandas as pd
import cv2
from functools import reduce
from sklearn.utils import shuffle
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import progressbar
from utils.transforms import Resize

classes = (
    "Supine",
    "Lateral_Right",
    "Lateral_Left",
    "KneeChest_Right",
    "KneeChest_Left",
    "Supine Bed Incline",
    "Right Body Roll",
    "Left Body Roll",
    "SittingOnEdge",
    "SittingOnBed",
    "Prone",
)


class PhysionetDataset(Dataset):
    labels_for_file = [0, 1, 2, 6, 6, 7, 7, 0, 0, 0, 0, 0, 3, 4, 5, 5, 5]
    data_folder: Path = Path("data").joinpath("physionet")
    classes2 = (
        "Supine",
        "Right",
        "Left",
        "Right Fetus",
        "Left Fetus",
        "Supine Bed Incline",
        "Right Body Roll",
        "Left Body Roll",
    )

    def __init__(self, transform=None, train=False):
        subjects = range(1, 9) if train else range(9, 14)
        X, Y = [], []
        for subject in subjects:
            data_file, labels_file = self.files_for_subject(subject)
            X.append(np.load(data_file))
            Y.append(np.load(labels_file))

        self.x, self.y = np.concatenate(X), np.concatenate(Y)

        self.n_samples = self.x.shape[0]

        self.transform = transform

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.n_samples

    @classmethod
    def files_for_subject(cls, subject: int):
        return cls.data_folder.joinpath(
            f"data_{subject}.npy"
        ), cls.data_folder.joinpath(f"labels_{subject}.npy")

    @classmethod
    def reload_data_from_source(cls):
        subjects = range(1, 14)
        records_per_subject = range(1, 18)
        raw_data_folder: Path = Path("data").joinpath("physionet-raw")
        widgets = [
            "Reading Files: ",
            progressbar.Bar(left="[", right="]", marker="-"),
            " ",
            progressbar.Counter(format="%(value)d/%(max_value)d"),
        ]

        with progressbar.ProgressBar(
            max_value=len(subjects) * len(records_per_subject), widgets=widgets
        ) as bar:
            for subject in subjects:
                x, y = [], []
                for file in records_per_subject:
                    # usecols makes sure that last column is skipped, skiprows is used to select which frame(s) are read
                    raw_frames = np.loadtxt(
                        raw_data_folder.joinpath(f"experiment-i/S{subject}/{file}.txt"),
                        delimiter="\t",
                        usecols=([_ for _ in range(2048)]),
                        skiprows=2,
                        dtype=np.float32,
                    )

                    raw_frames = np.reshape(raw_frames, (-1, 64, 32))
                    x.append(raw_frames)
                    y.append(
                        np.full(
                            [raw_frames.shape[0]],
                            cls.labels_for_file[file - 1],
                        )
                    )
                    bar.update(
                        ((subject - subjects[0]) * len(records_per_subject)) + file,
                    )
                x = np.concatenate(x, axis=0)
                y = np.concatenate(y, axis=0)

                cls.data_folder.mkdir(parents=True, exist_ok=True)
                data_file, labels_file = cls.files_for_subject(subject)
                np.save(data_file, x)
                np.save(labels_file, y)


class AmbientaDataset(Dataset):
    directory = "./data/ambienta/"
    classes2 = [
        "Supine",
        "SittingOnEdge",
        "SittingOnBed",
        "Lateral_Right",
        "Prone",
        "Lateral_Left",
        "KneeChest_Left",
    ]

    def __init__(self, transform=None, train=False):
        x_arrays = []
        y_arrays = []
        subjects = range(3, 4) if train else range(5, 6)
        for subject in subjects:
            raw_frames = np.loadtxt(
                f"{self.directory}{subject}.gz", delimiter=",", dtype=np.float32
            )
            raw_frames = np.reshape(raw_frames, (-1, 1, 64, 26))

            raw_labels = pd.read_csv(f"{self.directory}{subject}_labels.csv")

            labels = []
            frames_to_remove = []
            for frame_nr, _, label in raw_labels.itertuples():
                if label in classes:
                    labels.append(classes.index(label))
                else:
                    frames_to_remove.append(frame_nr)
                    labels.append(-1)

            raw_frames = np.delete(raw_frames, frames_to_remove, 0)
            labels = np.delete(labels, frames_to_remove, 0)

            x_arrays.append(raw_frames)
            y_arrays.append(labels)

        self.x = np.concatenate(x_arrays)
        self.y = np.concatenate(y_arrays)
        # self.x, self.y = shuffle(self.x, self.y, random_state=234950)
        self.n_samples = self.x.shape[0]

        self.transform = transform

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.n_samples


class SLPDataset(Dataset):
    data_folder: Path = Path("data").joinpath("SLP")

    def __init__(self, transform=None, train=False):
        data_file, labels_file = self.files()
        self.x = np.load(data_file)
        self.y = np.load(labels_file)
        # print(self.x.shape, self.y.shape)
        self.x = np.reshape(self.x, (-1, *self.x.shape[2:]))
        self.y = np.reshape(self.y, (-1, *self.y.shape[2:]))
        # print(self.x.shape, self.y.shape)

        self.n_samples = self.x.shape[0]

        self.transform = transform

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.n_samples

    @classmethod
    def files(cls):
        return cls.data_folder.joinpath(f"data.npy"), cls.data_folder.joinpath(
            f"labels.npy"
        )

    @classmethod
    def reload_data_from_source(cls):
        subjects = range(1, 103)
        covers = ["cover1", "cover2", "uncover"]
        raw_data_folder: Path = Path("data").joinpath("SLP-raw").joinpath("danaLab")
        widgets = [
            "Reading Files: ",
            progressbar.Bar(left="[", right="]", marker="-"),
            " ",
            progressbar.Counter(format="%(value)d/%(max_value)d"),
        ]

        with progressbar.ProgressBar(
            max_value=len(subjects) * 135, widgets=widgets
        ) as bar:
            data = []
            labels = []  # 0 = supine, 1 = left, 2 = right
            for subject in subjects:
                data.append([])
                labels.append([])
                for cover in covers:
                    for pos in range(1, 46):
                        file_data = np.load(
                            raw_data_folder.joinpath(f"00{str(subject).zfill(3)}")
                            .joinpath("PMarray")
                            .joinpath(cover)
                            .joinpath(f"0000{str(pos).zfill(2)}.npy")
                        )
                        data[subject - 1].append(file_data)
                        if pos <= 15:
                            labels[subject - 1].append(0)
                        elif pos <= 30:
                            labels[subject - 1].append(1)
                        else:
                            labels[subject - 1].append(2)
                        bar.update(
                            ((subject - subjects[0]) * 135)
                            + covers.index(cover) * 45
                            + pos,
                        )
            data = np.asarray(data)
            labels = np.asarray(labels)
            cls.data_folder.mkdir(parents=True, exist_ok=True)
            data_file, labels_file = cls.files()
            np.save(data_file, data)
            np.save(labels_file, labels)
