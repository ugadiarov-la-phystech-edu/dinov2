# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import csv
from enum import Enum
import logging
import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from .extended import ExtendedVisionDataset


logger = logging.getLogger("dinov2")
_Target = int


class _Split(Enum):
    TRAIN = "train"
    VAL = "val"


    @property
    def length(self) -> int:
        split_lengths = {
            _Split.TRAIN: 1607400,
            _Split.VAL: 402160,
        }
        return split_lengths[self]

    def get_dirname(self, class_id: Optional[str] = None) -> str:
        return self.value if class_id is None else os.path.join(self.value, class_id)


class Synthetic(ExtendedVisionDataset):
    Target = Union[_Target]
    Split = Union[_Split]

    def __init__(
        self,
        *,
        split: "Synthetic.Split",
        root: str,
        extra: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._extra_root = extra
        self._split = split

        self._entries = None
        self._class_ids = None
        self._class_names = None

    @property
    def split(self) -> "Synthetic.Split":
        return self._split

    def _get_extra_full_path(self, extra_path: str) -> str:
        return os.path.join(self._extra_root, extra_path)

    def _load_extra(self, extra_path: str) -> np.ndarray:
        extra_full_path = self._get_extra_full_path(extra_path)
        return np.load(extra_full_path, mmap_mode="r")

    def _save_extra(self, extra_array: np.ndarray, extra_path: str) -> None:
        extra_full_path = self._get_extra_full_path(extra_path)
        os.makedirs(self._extra_root, exist_ok=True)
        np.save(extra_full_path, extra_array)

    @property
    def _entries_path(self) -> str:
        return f"entries-{self._split.value.upper()}.npy"

    @property
    def _class_ids_path(self) -> str:
        return f"class-ids-{self._split.value.upper()}.npy"

    @property
    def _class_names_path(self) -> str:
        return f"class-names-{self._split.value.upper()}.npy"

    def _get_entries(self) -> np.ndarray:
        if self._entries is None:
            self._entries = self._load_extra(self._entries_path)
        assert self._entries is not None
        return self._entries

    def _get_class_ids(self) -> np.ndarray:
        if self._class_ids is None:
            self._class_ids = self._load_extra(self._class_ids_path)
        assert self._class_ids is not None
        return self._class_ids

    def _get_class_names(self) -> np.ndarray:
        # if self._split == _Split.TEST:
        #     assert False, "Class names are not available in TEST split"
        if self._class_names is None:
            self._class_names = self._load_extra(self._class_names_path)
        assert self._class_names is not None
        return self._class_names

    def find_class_id(self, class_index: int) -> str:
        class_ids = self._get_class_ids()
        return str(class_ids[class_index])

    def find_class_name(self, class_index: int) -> str:
        class_names = self._get_class_names()
        return str(class_names[class_index])

    def get_image_data(self, index: int) -> bytes:
        entries = self._get_entries()
        image_relpath = entries[index]["relpath"]
        image_full_path = os.path.join(self.root, image_relpath)
        with open(image_full_path, mode="rb") as f:
            image_data = f.read()
        return image_data

    def get_target(self, index: int) -> Optional[Target]:
        entries = self._get_entries()
        class_index = entries[index]["class_index"]
        return int(class_index)

    def get_targets(self) -> Optional[np.ndarray]:
        entries = self._get_entries()
        return entries["class_index"]

    def get_class_id(self, index: int) -> Optional[str]:
        entries = self._get_entries()
        class_id = entries[index]["class_id"]
        return str(class_id)
        # return None if self.split == _Split.TEST else str(class_id)

    def get_class_name(self, index: int) -> Optional[str]:
        entries = self._get_entries()
        class_name = entries[index]["class_name"]
        return str(class_name)
        # return None if self.split == _Split.TEST else str(class_name)

    def __len__(self) -> int:
        entries = self._get_entries()
        assert len(entries) == self.split.length
        return len(entries)
    
    def _load_labels(self) -> List[Tuple[str, str]]:
        labels = []
        dataset_root = os.path.join(self.root, self._split.get_dirname())
        for dir in os.listdir(dataset_root):
            labels.append((dir, dir))
        return labels

    def _dump_entries(self) -> None:
        split = self.split
        labels = self._load_labels()

        # NOTE: Using torchvision ImageFolder for consistency
        from torchvision.datasets import ImageFolder

        dataset_root = os.path.join(self.root, split.get_dirname())
        dataset = ImageFolder(dataset_root)
        sample_count = len(dataset)
        max_class_id_length, max_class_name_length, max_image_relpath_length = -1, -1, -1
        for sample in dataset.samples:
            image_fullpath, class_index = sample
            image_relpath = os.path.relpath(image_fullpath, self.root)
            class_id, class_name = labels[class_index]
            max_class_id_length = max(len(class_id), max_class_id_length)
            max_class_name_length = max(len(class_name), max_class_name_length)
            max_image_relpath_length = max(len(image_relpath), max_image_relpath_length)

        dtype = np.dtype(
            [
                ("class_index", "<u4"),
                ("class_id", f"U{max_class_id_length}"),
                ("class_name", f"U{max_class_name_length}"),
                ("relpath", f"U{max_image_relpath_length}")
            ]
        )
        entries_array = np.empty(sample_count, dtype=dtype)
        
        class_names = {class_id: class_name for class_id, class_name in labels}

        assert dataset
        old_percent = -1
        for index in range(sample_count):
            percent = 100 * (index + 1) // sample_count
            if percent > old_percent:
                logger.info(f"creating entries: {percent}%")
                old_percent = percent

            image_fullpath, class_index = dataset.samples[index]
            image_relpath = os.path.relpath(image_fullpath, self.root)
            class_id = Path(image_relpath).parts[1]
            class_name = class_names[class_id]
            entries_array[index] = (class_index, class_id, class_name, image_relpath)

        logger.info(f'saving entries to "{self._entries_path}"')
        self._save_extra(entries_array, self._entries_path)

    def _dump_class_ids_and_names(self) -> None:

        entries_array = self._load_extra(self._entries_path)

        max_class_id_length, max_class_name_length, max_class_index = -1, -1, -1
        for entry in entries_array:
            class_index, class_id, class_name = (
                entry["class_index"],
                entry["class_id"],
                entry["class_name"]
            )
            max_class_index = max(int(class_index), max_class_index)
            max_class_id_length = max(len(str(class_id)), max_class_id_length)
            max_class_name_length = max(len(str(class_name)), max_class_name_length)

        class_count = max_class_index + 1
        class_ids_array = np.empty(class_count, dtype=f"U{max_class_id_length}")
        class_names_array = np.empty(class_count, dtype=f"U{max_class_name_length}")
        for entry in entries_array:
            class_index, class_id, class_name = (
                entry["class_index"],
                entry["class_id"],
                entry["class_name"],
            )
            class_ids_array[class_index] = class_id
            class_names_array[class_index] = class_name

        logger.info(f'saving class IDs to "{self._class_ids_path}"')
        self._save_extra(class_ids_array, self._class_ids_path)

        logger.info(f'saving class names to "{self._class_names_path}"')
        self._save_extra(class_names_array, self._class_names_path)

    def dump_extra(self) -> None:
        self._dump_entries()
        self._dump_class_ids_and_names()
