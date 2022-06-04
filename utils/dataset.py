from enum import IntEnum
from pathlib import Path

import numpy as np
import pandas as pd
import progressbar
from torch.utils.data import Dataset


class PostureClass(IntEnum):
    SUPINE = 0
    LEFT = 1
    RIGHT = 2

    def __str__(self) -> str:
        return self.name.capitalize()


class PhysionetDataset(Dataset):
    data_folder: Path = Path("data").joinpath("physionet")

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
        image = self.x[index]
        if self.transform:
            image = self.transform(image)
        return image, self.y[index]

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
        # records_per_subject.remove(5)
        # records_per_subject.remove(7)
        labels_for_file = [
            PostureClass.SUPINE,
            PostureClass.LEFT,
            PostureClass.RIGHT,
            PostureClass.LEFT,
            PostureClass.LEFT,
            PostureClass.RIGHT,
            PostureClass.RIGHT,
            PostureClass.SUPINE,
            PostureClass.SUPINE,
            PostureClass.SUPINE,
            PostureClass.SUPINE,
            PostureClass.SUPINE,
            PostureClass.LEFT,
            PostureClass.RIGHT,
            PostureClass.SUPINE,
            PostureClass.SUPINE,
            PostureClass.SUPINE,
        ]
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
                            labels_for_file[file - 1],
                        )
                    )
                    bar.update(
                        ((subject - subjects[0]) * len(records_per_subject)) + labels_for_file.index(file),
                    )
                x = np.concatenate(x, axis=0)
                y = np.concatenate(y, axis=0)

                cls.data_folder.mkdir(parents=True, exist_ok=True)
                data_file, labels_file = cls.files_for_subject(subject)
                np.save(data_file, x)
                np.save(labels_file, y)


class AmbientaDataset(Dataset):
    data_folder = Path("data").joinpath("ambienta")

    def __init__(self, transform=None, train=False):
        X, Y = [], []
        subjects = range(3, 5) if train else range(5, 6)
        for subject in subjects:
            data_file, labels_file = self.files_for_subject(subject)
            X.append(np.load(data_file))
            Y.append(np.load(labels_file))

        self.x, self.y = np.concatenate(X), np.concatenate(Y)

        self.n_samples = self.x.shape[0]

        self.transform = transform

    def __getitem__(self, index):
        image = np.flipud(self.x[index])
        if self.transform:
            image = self.transform(image)
        return image, self.y[index]

    def __len__(self):
        return self.n_samples

    @classmethod
    def files_for_subject(cls, subject):
        return cls.data_folder.joinpath(f"data_{subject}.npy"), cls.data_folder.joinpath(
            f"labels_{subject}.npy"
        )

    @classmethod
    def reload_data_from_source(cls):
        raw_data_folder = Path("data").joinpath("ambienta-raw")
        subjects = range(3, 6)
        label_for_class = {
            "Supine": PostureClass.SUPINE,
            "Lateral_Right": PostureClass.RIGHT,
            "Lateral_Left": PostureClass.LEFT,
            "KneeChest_Right": PostureClass.RIGHT,
            "KneeChest_Left": PostureClass.LEFT,
        }
        for subject in subjects:
            frames = np.loadtxt(
                raw_data_folder.joinpath(f"{subject}.gz"), delimiter=",", dtype=np.float32
            )
            frames = np.reshape(frames, (-1, 64, 26))

            raw_labels = pd.read_csv(raw_data_folder.joinpath(f"{subject}_labels.csv"))

            labels = []
            frames_to_remove = []
            for frame_nr, _, label in raw_labels.itertuples():
                if label in label_for_class:
                    labels.append(label_for_class[label])
                else:
                    frames_to_remove.append(frame_nr)
                    labels.append(-1)

            frames = np.delete(frames, frames_to_remove, 0)
            labels = np.delete(labels, frames_to_remove, 0)

            cls.data_folder.mkdir(parents=True, exist_ok=True)
            data_file, labels_file = cls.files_for_subject(subject)
            np.save(data_file, frames)
            np.save(labels_file, labels)


class SLPDataset(Dataset):
    data_folder: Path = Path("data").joinpath("SLP")

    def __init__(self, transform=None, train=False):
        subject_bottom, subject_top = (0, 80) if train else (80, 102)
        data_file, labels_file = self.files()
        self.x = np.load(data_file)[subject_bottom:subject_top]
        self.y = np.load(labels_file)[subject_bottom:subject_top]
        # print(self.x.shape, self.y.shape)
        self.x = np.reshape(self.x, (-1, *self.x.shape[2:]))
        self.y = np.reshape(self.y, (-1, *self.y.shape[2:]))
        # print(self.x.shape, self.y.shape)

        self.n_samples = self.x.shape[0]

        self.transform = transform

    def __getitem__(self, index):
        image = self.x[index]
        if self.transform:
            image = self.transform(image)
        return image, self.y[index]

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
                            labels[subject - 1].append(PostureClass.SUPINE)
                        elif pos <= 30:
                            labels[subject - 1].append(PostureClass.LEFT)
                        else:
                            labels[subject - 1].append(PostureClass.RIGHT)
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
