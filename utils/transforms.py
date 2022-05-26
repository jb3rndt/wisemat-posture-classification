from typing import Callable, Tuple
import torch
import cv2
import math
import numpy as np
from skimage import restoration, filters

from utils.visualizations import apply_lines


class ToTensor:
    def __call__(self, sample):
        image, label = sample
        return torch.from_numpy(image), label

    def __repr__(self):
        return "ToTensor"


class NormalizeMean:
    def __call__(self, sample):
        image, label = sample
        return image - np.mean(image), label

    def __repr__(self):
        return "MeanNormalization"


class Standardize:
    def __call__(self, sample):
        image, label = sample
        return image / np.std(image), label

    def __repr__(self):
        return "Standardization"


class NormalizeValues:
    def __init__(self, vmin=0, vmax=1):
        assert vmin < vmax, "vmin must be smaller than vmax"
        self.vmin = vmin
        self.vmax = vmax

    def __call__(self, sample):
        image, label = sample
        image /= np.max(image)
        return image * (self.vmax - self.vmin) + self.vmin, label

    def __repr__(self):
        return f"Normalize: {self.vmin} - {self.vmax}"


class Blur:
    def __init__(self, ksize) -> None:
        self.ksize = ksize

    def __call__(self, sample):
        image, label = sample
        frame_blur = cv2.GaussianBlur(image, self.ksize, cv2.BORDER_DEFAULT)
        return frame_blur, label

    def __repr__(self):
        return f"GaussianBlur: kernel={self.ksize}"


class Laplace:
    def __call__(self, sample):
        image, label = sample
        frame_laplace = cv2.Laplacian(image, cv2.CV_32F)
        return (
            image + frame_laplace,
            label,
        )  # Adding laplace image to the original: https://towardsdatascience.com/image-filters-in-python-26ee938e57d2

    def __repr__(self):
        return "Laplace"


class Threshold:
    def __init__(
        self,
        threshold_fn: Callable[[np.ndarray], float],
        maxval=1,
        type=cv2.THRESH_TOZERO,
    ) -> None:
        self.threshold_fn = threshold_fn
        self.maxval = maxval
        self.type = type

    def __call__(self, sample: Tuple[np.ndarray, np.ndarray]):
        image, label = sample
        th, frame_thresh = cv2.threshold(
            image, self.threshold_fn(image), self.maxval, self.type
        )
        return frame_thresh, label

    def __repr__(self):
        return f"Threshold: maxval={self.maxval}, type={self.type}"


class Erode:
    def __init__(self, ksize=(3, 3), iterations=1, ktype=cv2.MORPH_RECT) -> None:
        self.ksize = ksize
        self.ktype = ktype
        self.iterations = iterations

    def __call__(self, sample):
        image, label = sample
        kernel = cv2.getStructuringElement(self.ktype, self.ksize)
        frame_eroded = cv2.erode(image, kernel, iterations=self.iterations)
        return frame_eroded, label

    def __repr__(self):
        return f"Erode: kernel={self.ksize}, iterations={self.iterations}, kernel_type={self.ktype}"


class Dilate:
    def __init__(self, ksize=(3, 3), iterations=1, ktype=cv2.MORPH_RECT) -> None:
        self.ksize = ksize
        self.ktype = ktype
        self.iterations = iterations

    def __call__(self, sample):
        image, label = sample
        kernel = cv2.getStructuringElement(self.ktype, self.ksize)
        frame_eroded = cv2.dilate(image, kernel, iterations=self.iterations)
        return frame_eroded, label

    def __repr__(self):
        return f"Dilate: kernel={self.ksize}, iterations={self.iterations}, kernel_type={self.ktype}"


class Close:
    def __init__(
        self, ksize=(3, 3), iterations=1, ktype=cv2.MORPH_RECT, kernel=None
    ) -> None:
        self.ksize = ksize
        self.ktype = ktype
        self.iterations = iterations
        self.kernel = (
            cv2.getStructuringElement(self.ktype, self.ksize)
            if kernel is None
            else kernel
        )

    def __call__(self, sample):
        image, label = sample
        frame_closed = cv2.morphologyEx(
            image, cv2.MORPH_CLOSE, self.kernel, iterations=self.iterations
        )
        return frame_closed, label

    def __repr__(self):
        return f"Close: kernel={self.ksize}, iterations={self.iterations}, kernel_type={self.ktype}"


class Open:
    def __init__(self, ksize=(3, 3), iterations=1, ktype=cv2.MORPH_RECT) -> None:
        self.ksize = ksize
        self.ktype = ktype
        self.iterations = iterations

    def __call__(self, sample):
        image, label = sample
        kernel = cv2.getStructuringElement(self.ktype, self.ksize)
        frame_closed = cv2.morphologyEx(
            image, cv2.MORPH_OPEN, kernel, iterations=self.iterations
        )
        return frame_closed, label

    def __repr__(self):
        return f"Open: kernel={self.ksize}, iterations={self.iterations}, kernel_type={self.ktype}"


class Resize:
    def __init__(self, size, interpolation=cv2.INTER_LINEAR) -> None:
        self.size = size
        self.interpolation = interpolation

    def __call__(self, sample):
        image, label = sample
        frame_resized = cv2.resize(image, self.size, interpolation=self.interpolation)
        return frame_resized, label

    def __repr__(self):
        return f"Resize: size={self.size}, interpolation={self.interpolation}"


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


def roll_ball(image, radius=100, normalized=True):
    if normalized:
        kernel = restoration.ellipsoid_kernel(
            tuple([radius * 2] * image.ndim), radius / 255 * 2
        )
        return restoration.rolling_ball(image, kernel=kernel)
    return restoration.rolling_ball(image, radius=radius)


class RollingBall:
    def __init__(self, radius=100, normalized=True) -> None:
        self.radius = radius
        self.normalized = normalized

    def __call__(self, sample):
        image, label = sample
        return (
            image - roll_ball(image, radius=self.radius, normalized=self.normalized),
            label,
        )

    def __repr__(self):
        return f"RollingBall: radius={self.radius}, normalized={self.normalized}"


def low_pass(img, rad=60):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    crow, ccol = tuple(d // 2 for d in img.shape)
    low_pass_filter = np.zeros(fshift.shape)
    low_pass_filter[crow - rad : crow + rad, ccol - rad : ccol + rad] = 1
    fshift *= low_pass_filter
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    return np.abs(img_back)


def high_pass(img, rad=60):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    crow, ccol = tuple(d // 2 for d in img.shape)
    fshift[crow - rad : crow + rad, ccol - rad : ccol + rad] = 0
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    return np.abs(img_back)


class HighPass:
    def __init__(self, rad=60):
        self.rad = rad

    def __call__(self, sample):
        image, label = sample
        return high_pass(image, self.rad), label

    def __repr__(self):
        return f"HighPass: radius={self.rad}"


class LowPass:
    def __init__(self, rad=60):
        self.rad = rad

    def __call__(self, sample):
        image, label = sample
        return low_pass(image, self.rad), label

    def __repr__(self):
        return f"LowPass: radius={self.rad}"


class Sobel:
    def __call__(self, sample):
        image, label = sample
        return filters.sobel(image), label

    def __repr__(self):
        return "Sobel"


class EqualizeHist:
    def __call__(self, sample):
        image, label = sample
        image = image * 255.0
        image = image.astype(np.uint8)
        image = cv2.equalizeHist(image)
        image = image.astype(np.float32) / 255.0
        return image, label

    def __repr__(self):
        return "EqualizeHist"


class Denoise:
    def __call__(self, sample):
        image, label = sample
        img = np.uint8(image * 255)
        img = cv2.fastNlMeansDenoising(img, None, 9, 7, 21)
        return img, label

    def __repr__(self):
        return "Denoise"


def hough_lines(image):
    sobel, __ = Sobel()((image, 0))
    sobel, __ = Threshold(
        lambda img: np.median(img[img > 0.0]), type=cv2.THRESH_BINARY
    )((sobel, 0))
    sobel = np.uint8(sobel * 255)
    lines = cv2.HoughLines(sobel, 1, np.pi / 180, 10)
    lines = lines[:, 0, :] if lines is not None else np.empty((0, 2))
    return lines, sobel


class CloseInHoughDirection:
    def __init__(self, debug_lines=False):
        self.debug_lines = debug_lines

    def __call__(self, sample):
        image, label = sample
        lines, __ = hough_lines(image)
        if len(lines) == 0:
            print("No lines found. Not applying closing.")
            return image, label
        kernel = choose_kernel_ext(math.degrees(lines[0][1]))
        closed, label = Close(kernel=kernel)((image, label))
        if self.debug_lines:
            closed = apply_lines(closed, lines)
        return closed, label

    def __repr__(self):
        return "CloseInHoughDirection"

# try all threshold
class PouyanProcessing:
    def __call__(self, sample):
        image, label = sample
        denoised = filters.median(image)
        __, binary = cv2.threshold(denoised, 0, 1, cv2.THRESH_BINARY)
        cv2.floodFill(binary, None, (0, 0), 1)
        return binary, label


def is_vertical(deg):
    return (
        (deg >= 0 and deg < 22.5)
        or (deg >= 180 - 22.5 and deg < 180 + 22.5)
        or (deg >= 360 - 22.5 and deg < 360)
    )


def is_sw_ne(deg):
    return (deg >= 45 - 22.5 and deg < 45 + 22.5) or (
        deg >= 225 - 22.5 and deg < 225 + 22.5
    )


def is_horizontal(deg):
    return (deg >= 90 - 22.5 and deg < 90 + 22.5) or (
        deg >= 270 - 22.5 and deg < 270 + 22.5
    )


def is_se_nw(deg):
    return (deg >= 135 - 22.5 and deg < 135 + 22.5) or (
        deg >= 315 - 22.5 and deg < 315 + 22.5
    )


def choose_kernel_ext(deg):
    if is_vertical(deg):
        return np.array(
            [
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
            ],
            dtype=np.uint8,
        )
    if is_sw_ne(deg):
        return np.array(
            [
                [0, 0, 0, 1, 1],
                [0, 0, 1, 1, 1],
                [0, 1, 1, 1, 0],
                [1, 1, 1, 0, 0],
                [1, 1, 0, 0, 0],
            ],
            dtype=np.uint8,
        )
    if is_horizontal(deg):
        return np.array(
            [
                [0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.uint8,
        )
    if is_se_nw(deg):
        return np.array(
            [
                [1, 1, 0, 0, 0],
                [1, 1, 1, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 1, 1, 1],
                [0, 0, 0, 1, 1],
            ],
            dtype=np.uint8,
        )
    raise Exception(f"Unknown degree: {deg}")


# Code from https://github.com/m4nv1r/medium_articles/blob/master/Image_Filters_in_Python.ipynb
def crimmins(data):
    new_image = data.copy()
    nrow = len(data)
    ncol = len(data[0])

    # Dark pixel adjustment

    # First Step
    # N-S
    for i in range(1, nrow):
        for j in range(ncol):
            if data[i - 1, j] >= (data[i, j] + 2):
                new_image[i, j] += 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(ncol - 1):
            if data[i, j + 1] >= (data[i, j] + 2):
                new_image[i, j] += 1
    data = new_image
    # NW-SE
    for i in range(1, nrow):
        for j in range(1, ncol):
            if data[i - 1, j - 1] >= (data[i, j] + 2):
                new_image[i, j] += 1
    data = new_image
    # NE-SW
    for i in range(1, nrow):
        for j in range(ncol - 1):
            if data[i - 1, j + 1] >= (data[i, j] + 2):
                new_image[i, j] += 1
    data = new_image
    # Second Step
    # N-S
    for i in range(1, nrow - 1):
        for j in range(ncol):
            if (data[i - 1, j] > data[i, j]) and (data[i, j] <= data[i + 1, j]):
                new_image[i, j] += 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(1, ncol - 1):
            if (data[i, j + 1] > data[i, j]) and (data[i, j] <= data[i, j - 1]):
                new_image[i, j] += 1
    data = new_image
    # NW-SE
    for i in range(1, nrow - 1):
        for j in range(1, ncol - 1):
            if (data[i - 1, j - 1] > data[i, j]) and (data[i, j] <= data[i + 1, j + 1]):
                new_image[i, j] += 1
    data = new_image
    # NE-SW
    for i in range(1, nrow - 1):
        for j in range(1, ncol - 1):
            if (data[i - 1, j + 1] > data[i, j]) and (data[i, j] <= data[i + 1, j - 1]):
                new_image[i, j] += 1
    data = new_image
    # Third Step
    # N-S
    for i in range(1, nrow - 1):
        for j in range(ncol):
            if (data[i + 1, j] > data[i, j]) and (data[i, j] <= data[i - 1, j]):
                new_image[i, j] += 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(1, ncol - 1):
            if (data[i, j - 1] > data[i, j]) and (data[i, j] <= data[i, j + 1]):
                new_image[i, j] += 1
    data = new_image
    # NW-SE
    for i in range(1, nrow - 1):
        for j in range(1, ncol - 1):
            if (data[i + 1, j + 1] > data[i, j]) and (data[i, j] <= data[i - 1, j - 1]):
                new_image[i, j] += 1
    data = new_image
    # NE-SW
    for i in range(1, nrow - 1):
        for j in range(1, ncol - 1):
            if (data[i + 1, j - 1] > data[i, j]) and (data[i, j] <= data[i - 1, j + 1]):
                new_image[i, j] += 1
    data = new_image
    # Fourth Step
    # N-S
    for i in range(nrow - 1):
        for j in range(ncol):
            if data[i + 1, j] >= (data[i, j] + 2):
                new_image[i, j] += 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(1, ncol):
            if data[i, j - 1] >= (data[i, j] + 2):
                new_image[i, j] += 1
    data = new_image
    # NW-SE
    for i in range(nrow - 1):
        for j in range(ncol - 1):
            if data[i + 1, j + 1] >= (data[i, j] + 2):
                new_image[i, j] += 1
    data = new_image
    # NE-SW
    for i in range(nrow - 1):
        for j in range(1, ncol):
            if data[i + 1, j - 1] >= (data[i, j] + 2):
                new_image[i, j] += 1
    data = new_image

    # Light pixel adjustment

    # First Step
    # N-S
    for i in range(1, nrow):
        for j in range(ncol):
            if data[i - 1, j] <= (data[i, j] - 2):
                new_image[i, j] -= 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(ncol - 1):
            if data[i, j + 1] <= (data[i, j] - 2):
                new_image[i, j] -= 1
    data = new_image
    # NW-SE
    for i in range(1, nrow):
        for j in range(1, ncol):
            if data[i - 1, j - 1] <= (data[i, j] - 2):
                new_image[i, j] -= 1
    data = new_image
    # NE-SW
    for i in range(1, nrow):
        for j in range(ncol - 1):
            if data[i - 1, j + 1] <= (data[i, j] - 2):
                new_image[i, j] -= 1
    data = new_image
    # Second Step
    # N-S
    for i in range(1, nrow - 1):
        for j in range(ncol):
            if (data[i - 1, j] < data[i, j]) and (data[i, j] >= data[i + 1, j]):
                new_image[i, j] -= 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(1, ncol - 1):
            if (data[i, j + 1] < data[i, j]) and (data[i, j] >= data[i, j - 1]):
                new_image[i, j] -= 1
    data = new_image
    # NW-SE
    for i in range(1, nrow - 1):
        for j in range(1, ncol - 1):
            if (data[i - 1, j - 1] < data[i, j]) and (data[i, j] >= data[i + 1, j + 1]):
                new_image[i, j] -= 1
    data = new_image
    # NE-SW
    for i in range(1, nrow - 1):
        for j in range(1, ncol - 1):
            if (data[i - 1, j + 1] < data[i, j]) and (data[i, j] >= data[i + 1, j - 1]):
                new_image[i, j] -= 1
    data = new_image
    # Third Step
    # N-S
    for i in range(1, nrow - 1):
        for j in range(ncol):
            if (data[i + 1, j] < data[i, j]) and (data[i, j] >= data[i - 1, j]):
                new_image[i, j] -= 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(1, ncol - 1):
            if (data[i, j - 1] < data[i, j]) and (data[i, j] >= data[i, j + 1]):
                new_image[i, j] -= 1
    data = new_image
    # NW-SE
    for i in range(1, nrow - 1):
        for j in range(1, ncol - 1):
            if (data[i + 1, j + 1] < data[i, j]) and (data[i, j] >= data[i - 1, j - 1]):
                new_image[i, j] -= 1
    data = new_image
    # NE-SW
    for i in range(1, nrow - 1):
        for j in range(1, ncol - 1):
            if (data[i + 1, j - 1] < data[i, j]) and (data[i, j] >= data[i - 1, j + 1]):
                new_image[i, j] -= 1
    data = new_image
    # Fourth Step
    # N-S
    for i in range(nrow - 1):
        for j in range(ncol):
            if data[i + 1, j] <= (data[i, j] - 2):
                new_image[i, j] -= 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(1, ncol):
            if data[i, j - 1] <= (data[i, j] - 2):
                new_image[i, j] -= 1
    data = new_image
    # NW-SE
    for i in range(nrow - 1):
        for j in range(ncol - 1):
            if data[i + 1, j + 1] <= (data[i, j] - 2):
                new_image[i, j] -= 1
    data = new_image
    # NE-SW
    for i in range(nrow - 1):
        for j in range(1, ncol):
            if data[i + 1, j - 1] <= (data[i, j] - 2):
                new_image[i, j] -= 1
    data = new_image
    return new_image.copy()


class Crimmins:
    def __call__(self, sample):
        image, label = sample
        img = np.uint8(image * 255)
        return crimmins(img), label
