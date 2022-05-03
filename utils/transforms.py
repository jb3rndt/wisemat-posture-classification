from typing import Any, Callable, Tuple
import torch
import cv2
import numpy as np
import torchvision


class ToTensor:
    def __call__(self, sample):
        image, label = sample
        return torch.from_numpy(image), label


class Center:
    def __call__(self, sample):
        image, label = sample
        return image - np.mean(image), label


class Blur:
    def __init__(self, ksize) -> None:
        self.ksize = ksize

    def __call__(self, sample):
        image, label = sample
        frame_blur = cv2.GaussianBlur(image[0], self.ksize, cv2.BORDER_DEFAULT)
        return np.array([frame_blur]), label


class Threshold:
    def __init__(self, threshold_fn: Callable[[np.ndarray], float]) -> None:
        self.threshold_fn = threshold_fn

    def __call__(self, sample: Tuple[np.ndarray, np.ndarray]):
        image, label = sample
        th, frame_thresh = cv2.threshold(
            image, self.threshold_fn(image), 1, cv2.THRESH_TOZERO
        )
        return frame_thresh, label


class Erode:
    def __init__(self, ksize=(3, 3), iterations=1, ktype=cv2.MORPH_RECT) -> None:
        self.ksize = ksize
        self.ktype = ktype
        self.iterations = iterations

    def __call__(self, sample):
        image, label = sample
        kernel = cv2.getStructuringElement(self.ktype, self.ksize)
        frame_eroded = cv2.erode(image[0], kernel, iterations=self.iterations)
        return np.array([frame_eroded]), label


class Dilate:
    def __init__(self, ksize=(3, 3), iterations=1, ktype=cv2.MORPH_RECT) -> None:
        self.ksize = ksize
        self.ktype = ktype
        self.iterations = iterations

    def __call__(self, sample):
        image, label = sample
        kernel = cv2.getStructuringElement(self.ktype, self.ksize)
        frame_eroded = cv2.dilate(image[0], kernel, iterations=self.iterations)
        return np.array([frame_eroded]), label


class Close:
    def __init__(self, ksize=(3, 3), iterations=1, ktype=cv2.MORPH_RECT) -> None:
        self.ksize = ksize
        self.ktype = ktype
        self.iterations = iterations

    def __call__(self, sample):
        image, label = sample
        kernel = cv2.getStructuringElement(self.ktype, self.ksize)
        frame_closed = cv2.morphologyEx(
            image[0], cv2.MORPH_CLOSE, kernel, iterations=self.iterations
        )
        return np.array([frame_closed]), label


class Open:
    def __init__(self, ksize=(3, 3), iterations=1, ktype=cv2.MORPH_RECT) -> None:
        self.ksize = ksize
        self.ktype = ktype
        self.iterations = iterations

    def __call__(self, sample):
        image, label = sample
        kernel = cv2.getStructuringElement(self.ktype, self.ksize)
        frame_closed = cv2.morphologyEx(
            image[0], cv2.MORPH_OPEN, kernel, iterations=self.iterations
        )
        return np.array([frame_closed]), label


class Resize:
    def __init__(self, size, interpolation) -> None:
        self.size = size
        self.interpolation = interpolation

    def __call__(self, sample):
        image, label = sample
        frame_resized = cv2.resize(
            image[0], self.size, interpolation=self.interpolation
        )
        return np.array([frame_resized]), label


def zca(data, epsilon=1e-5):
    data = data[:1000]
    data_shape = data.shape
    X = data.reshape(data_shape[0], -1)
    X = X / (np.max(X) - np.min(X))
    X = X - np.mean(X, axis=0)
    # X = X / np.sqrt((X ** 2).sum(axis=1))[:,None]
    cov = np.cov(X, rowvar=False)
    U, S, _ = np.linalg.svd(cov)
    zca_matrix = np.dot(np.dot(U, np.diag(1.0 / np.sqrt(S + epsilon))), U.T)
    X_ZCA = np.dot(zca_matrix, X.T).T
    X_ZCA = (X_ZCA - X_ZCA.min()) / (X_ZCA.max() - X_ZCA.min())
    return X_ZCA.reshape(data_shape)


class Normalize:
    def __call__(self, sample):
        image, label = sample
        return image / np.max(image), label
        mean = np.mean(x, (1, 2))
        std = np.std(x, (1, 2))
        # print(mean, std)
        return (x - x.mean(axis=(0, 1, 2), keepdims=True)) / x.std(
            axis=(0, 1, 2), keepdims=True
        )
        return np.asarray(
            torchvision.transforms.Normalize(mean, std)(torch.from_numpy(image))
        )


class EqualizeHist:
    def __call__(self, sample):
        image, label = sample
        image = image * 255.0
        image = image.astype(np.uint8)
        image = cv2.equalizeHist(image[0])
        image = image.astype(np.float32) / 255.0
        return np.array([image]), label
