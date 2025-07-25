# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Streaming images and labels from datasets created with dataset_tool.py."""

import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib
import sklearn.datasets

try:
    import pyspng
except ImportError:
    pyspng = None

#----------------------------------------------------------------------------
# Abstract base class for datasets.

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
        cache       = False,    # Cache images in CPU memory?
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._cache = cache
        self._cached_images = dict() # {raw_idx: np.ndarray, ...}
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed % (1 << 31)).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        raw_idx = self._raw_idx[idx]
        image = self._cached_images.get(raw_idx, None)
        if image is None:
            image = self._load_raw_image(raw_idx)
            if self._cache:
                self._cached_images[raw_idx] = image
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype in [np.uint8, np.float32]  # Allow float32 for custom datasets
        if self._xflip[idx]:
            assert image.ndim == 3 or image.ndim == 4  # CHW or CHW1
            image = image[..., ::-1] if image.ndim == 3 else image[..., ::-1, :]
        return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) in [3, 4]  # CHW or CHW1
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) in [3, 4]  # CHW or CHW1
        assert self.image_shape[1] == self.image_shape[2] or self.image_shape[2] == 1
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------
# Dataset subclass that loads images recursively from the specified directory
# or ZIP file.

class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        use_pyspng      = True, # Use pyspng if available?
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._use_pyspng = use_pyspng
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if self._use_pyspng and pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

#----------------------------------------------------------------------------
# Dataset subclass for 2D point cloud datasets (SwissRoll, CheckerBoard, NGaussianMixtures).

class Custom2DDataset(Dataset):
    def __init__(self,
        path,                   # Path identifier (e.g., 'custom:swissroll'), ignored for data storage.
        resolution=2,           # Resolution of the dataset (not used here, but kept for compatibility).
        max_size=None,          # Artificially limit the size of the dataset. None = no limit.
        dataset_type='swissroll',  # Type of dataset: 'swissroll', 'checkerboard', or 'ngaussian'.
        n_samples=8000,         # Number of samples for the dataset.
        num_mixture=8,          # Number of mixtures for NGaussianMixtures.
        radius=8.0,             # Radius for NGaussianMixtures.
        sigma=1.0,              # Sigma for NGaussianMixtures.
        use_labels=False,       # Enable conditioning labels? False = no labels.
        xflip=False,            # Artificially double the dataset via x-flips.
        random_seed=0,          # Random seed for max_size.
        cache=True,             # Cache data in CPU memory?
    ):
        self._dataset_type = dataset_type
        name = dataset_type
        self._data = None

        # Initialize the appropriate dataset
        if dataset_type == 'swissroll':
            data = sklearn.datasets.make_swiss_roll(n_samples=n_samples, noise=1.0)[0]
            data = data.astype("float32")[:, [0, 2]]
            data /= 4.0
            data = torch.from_numpy(data).float()
            r = 4.5
            data1 = data.clone() + torch.tensor([-r, -r])
            data2 = data.clone() + torch.tensor([-r, r])
            data3 = data.clone() + torch.tensor([r, -r])
            data4 = data.clone() + torch.tensor([r, r])
            self._data = torch.cat([data, data1, data2, data3, data4], axis=0)
        elif dataset_type == 'checkerboard':
            x1 = np.random.rand(n_samples) * 4 - 2
            x2_ = np.random.rand(n_samples) - np.random.randint(0, 2, n_samples) * 2
            x2 = x2_ + (np.floor(x1) % 2)
            self._data = torch.from_numpy(np.concatenate([x1[:, None], x2[:, None]], 1) * 2).float()
        elif dataset_type == 'ngaussian':
            mix_probs = [1/num_mixture] * num_mixture
            std = torch.stack([torch.ones(2) * sigma for _ in range(len(mix_probs))], dim=0)
            mix_probs = torch.tensor(mix_probs)
            mix_idx = torch.multinomial(mix_probs, n_samples, replacement=True)
            thetas = np.linspace(0, 2 * np.pi, num_mixture, endpoint=False)
            xs = radius * np.sin(thetas, dtype=np.float32)
            ys = radius * np.cos(thetas, dtype=np.float32)
            center = np.vstack([xs, ys]).T
            center = torch.tensor(center)
            centers = center[mix_idx]
            stds = std[mix_idx]
            self._data = torch.randn_like(centers) * stds + centers
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        # Convert to numpy for compatibility with Dataset base class
        self._data = self._data.numpy().astype(np.float32)  # Shape [N, 2]
        raw_shape = [len(self._data), 1, 2, 1]  # [N, C=1, H=2, W=1]
        super().__init__(name=name, raw_shape=raw_shape, use_labels=use_labels, xflip=xflip, random_seed=random_seed, cache=cache)

    def _load_raw_image(self, raw_idx):
        # Return data as a "fake image" with shape [1, 2, 1] and dtype float32
        data = self._data[raw_idx]  # Shape [2]
        image = data.reshape(1, 2, 1)  # Shape [C=1, H=2, W=1]
        return image

    def _load_raw_labels(self):
        return None  # No labels for these datasets

    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)  # image is [1, 2, 1], float32
        return image, label