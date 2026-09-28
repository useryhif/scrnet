"""Datasets for XRD spectra and crystallographic labels.

Rows in the ``.npy`` files are positional: the label columns and the auxiliary feature
columns are addressed by index (see ``labels.py``). Three datasets are provided:

* :class:`XRDDataset` — plain classification at one symmetry level.
* :class:`SubXRDDataset` — training set for a single space-group expert, with balanced
  negative sampling so that in-range and out-of-range samples appear 1:1.
* :class:`SubXRDDatasetTest` — evaluation set for an expert; only in-range samples are kept
  and no negatives are mixed in.
"""

import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from labels import (
    CLS_COLUMN,
    COL_ABC,
    COL_ANGLES,
    COL_CS_EMBED,
    COL_LATTICE_EMBED,
    COL_POINT_GROUP_EMBED,
    COL_XRD,
    LABELS,
)

_MIN_SAMPLES = 256


class XRDDataset(Dataset):
    """Plain classification dataset for one symmetry level.

    Args:
        root_dir: path to the ``.npy`` file.
        cls: one of ``crystal_system``, ``lattice``, ``point_group``, ``space_group``.

    Yields ``(spectrum, label, raw_row)``. The spectrum is read from the last column.
    """

    def __init__(self, root_dir, cls):
        self.root_dir = root_dir
        self.data = []
        self.labels = []

        self.label_column = CLS_COLUMN[cls]
        self.label_map = {name: i for i, name in enumerate(LABELS[cls])}

        for sample in np.load(root_dir, allow_pickle=True):
            self.data.append(sample)
            self.labels.append(self.label_map[sample[self.label_column]])

    def __len__(self):
        return len(self.data)

    def get_label_map(self):
        return self.label_map

    def __getitem__(self, idx):
        spectrum = torch.tensor(self.data[idx][-1], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.int64)
        return spectrum, label, self.data[idx]


class _SubXRDDatasetBase(Dataset):
    """Shared logic for the per-expert datasets.

    Splits rows into those whose label falls inside ``[start, start + length)`` and the rest.
    ``self.data`` / ``self.labels`` / ``self.label_conf`` hold the in-range rows; the
    out-of-range rows are kept in ``self.others`` for subclasses to sample from.
    """

    def __init__(self, root_dir, cls, start, length):
        self.root_dir = root_dir

        self.label_column = CLS_COLUMN[cls]
        self.label_map = {name: i for i, name in enumerate(LABELS[cls])}
        self.classes = LABELS[cls]

        rows = list(np.load(root_dir, allow_pickle=True))

        self.data = []
        self.labels = []
        self.label_conf = []
        self.others = []
        self.others_labels = []

        for sample in rows:
            label = self.label_map[sample[self.label_column]]
            if start <= label < start + length:
                self.data.append(sample)
                self.labels.append(label)
                self.label_conf.append(1)
            else:
                self.others.append(sample)
                self.others_labels.append(label)

        # Full, unfiltered rows — used for class-count statistics.
        self.dataset = rows

    def _repeat_until_min_size(self):
        """Replicate very small groups so batches survive ``drop_last=True``."""
        if len(self.data) < _MIN_SAMPLES:
            factor = (_MIN_SAMPLES // len(self.data)) * 2
            self.data = self.data * factor
            self.labels = self.labels * factor
            self.label_conf = self.label_conf * factor

    def __len__(self):
        return len(self.data)

    def get_label_map(self):
        return self.label_map

    def get_labels(self):
        return self.classes

    def get_num_list(self):
        """Number of samples per class, indexed by class id."""
        return pd.DataFrame(self.dataset).value_counts(self.label_column).reindex(self.classes)

    def __getitem__(self, idx):
        row = self.data[idx]
        return (
            torch.tensor(row[COL_XRD], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.int64),
            torch.tensor(self.label_conf[idx], dtype=torch.int64),
            torch.tensor(row[COL_CS_EMBED], dtype=torch.float32),
            torch.tensor(row[COL_LATTICE_EMBED], dtype=torch.float32),
            torch.tensor(row[COL_POINT_GROUP_EMBED], dtype=torch.float32),
            row,
            torch.tensor(row[COL_ABC], dtype=torch.float32),
            torch.tensor(row[COL_ANGLES], dtype=torch.float32),
        )


class SubXRDDataset(_SubXRDDatasetBase):
    """Training set for one space-group expert.

    Adds as many out-of-range samples (confidence label 0) as there are in-range ones, so the
    expert learns to reject space groups outside its own crystal system.

    Yields ``(spectrum, label, label_conf, cs, lattice, pg, raw_row, abc, angles)``.
    """

    def __init__(self, root_dir, cls, start, length):
        super().__init__(root_dir, cls, start, length)

        # Sample negatives to match the in-range count, before any replication.
        n = len(self.data)
        picks = random.sample(range(len(self.others)), n)
        self.data = self.data + [self.others[i] for i in picks]
        self.labels = self.labels + [self.others_labels[i] for i in picks]
        self.label_conf = self.label_conf + [0] * n

        self._repeat_until_min_size()


class SubXRDDatasetTest(_SubXRDDatasetBase):
    """Evaluation set for one space-group expert.

    Keeps only in-range samples, with no negative sampling, so accuracy and confidence are
    measured on the expert's own classes.

    Yields ``(spectrum, label, label_conf, cs, lattice, pg, raw_row, abc, angles)``.
    """

    def __init__(self, root_dir, cls, start, length):
        super().__init__(root_dir, cls, start, length)
        self._repeat_until_min_size()
