from __future__ import annotations

import os
import glob
import random
from typing import List, Literal, Tuple, Optional, Dict, Union, Iterator
from dataclasses import dataclass
from collections import defaultdict

import cv2
import torch
import torch.nn.functional as F
import tqdm
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset
from shapely import Polygon
from openslide import OpenSlide
from openslide.deepzoom import DeepZoomGenerator

from rank_induction.data_models import Patch, Patches, BinaryLabels, Polygons
from rank_induction.patch_filter import AnnotationFilter
from rank_induction.misc import get_deepzoom_level

RANK_INDUCTION_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(RANK_INDUCTION_DIR)


def get_key(path: str) -> str:
    """파일 경로에서 확장자를 제외한 basename만 반환한다."""
    return os.path.splitext(os.path.basename(path))[0]


@dataclass
class DataPoint:
    """Element of Batch"""

    key: str
    h5_path: Optional[str] = None
    bag_label: Optional[torch.Tensor] = None
    instance_labels: Optional[torch.Tensor] = None

    @classmethod
    def from_path(cls, path: str) -> DataPoint:
        """h5 경로만 알고 있을 때 key까지 채워서 DataPoint를 만든다."""
        return cls(key=get_key(path), h5_path=path)

    def cal_instance_labels_from_wsi(
        self,
        patch_path: str,
        slide_path: str,
        polygons: Polygons,
        method: Literal["attention_induction", "rank_induction"],
        overlap_ratio: float = 0.05,
        inplace: bool = False,
    ) -> torch.Tensor:
        """Instance label을 계산"""
        patches: Patches = Patches.from_patch_h5(patch_path)
        annotation_filter = AnnotationFilter(polygons.data, overlap_ratio)

        instance_labels = list()
        if method == "rank_induction":
            for patch in patches:
                x1, y1, x2, y2 = (
                    patch.coordinates.x_min,
                    patch.coordinates.y_min,
                    patch.coordinates.x_max,
                    patch.coordinates.y_max,
                )
                polygon = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
                is_overlapped = not annotation_filter(polygon)
                instance_labels.append(1.0 if is_overlapped else 0.0)

            instance_labels = torch.tensor(instance_labels, dtype=torch.float32)

        elif method == "attention_induction":
            osr = OpenSlide(slide_path)
            w, h = osr.dimensions
            thumnail = np.array(osr.get_thumbnail((int(w / 16), int(h / 16))))
            gray_image = cv2.cvtColor(thumnail, cv2.COLOR_RGB2GRAY)
            threshold_val, _ = cv2.threshold(
                gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            osr.close()
            for patch in patches:
                query_polygon = patch.coordinates.to_polygon()
                # 주의: annotation_filter() True면 겹침 없음
                is_overlapped = not annotation_filter(query_polygon)
                if not is_overlapped:
                    instance_labels.append(0.0)
                    continue

                gray_patch = cv2.cvtColor(patch.image_array, cv2.COLOR_RGB2GRAY)
                n_tissue = float((gray_patch <= threshold_val).sum())
                instance_labels.append(n_tissue)

            total = np.sum(instance_labels)
            if total > 0.0:
                instance_labels = np.array(instance_labels) / total
            instance_labels = torch.tensor(instance_labels, dtype=torch.float32)

        if inplace:
            self.instance_labels = instance_labels
        return instance_labels

    def cal_instance_labels_from_image(
        self,
        patch_path: str,
        slide_path: str,
        polygons: Polygons,
        method: Literal["attention_induction", "rank_induction"],
        overlap_ratio: float = 0.05,
        inplace: bool = False,
    ) -> torch.Tensor:
        """Instance label을 계산"""
        patches: Patches = Patches.from_patch_h5(patch_path)
        annotation_filter = AnnotationFilter(polygons.data, overlap_ratio)

        instance_labels = list()
        if method == "rank_induction":
            for patch in patches:
                x1, y1, x2, y2 = (
                    patch.coordinates.x_min,
                    patch.coordinates.y_min,
                    patch.coordinates.x_max,
                    patch.coordinates.y_max,
                )
                polygon = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
                is_overlapped = not annotation_filter(polygon)
                instance_labels.append(1.0 if is_overlapped else 0.0)

            instance_labels = torch.tensor(instance_labels, dtype=torch.float32)

        elif method == "attention_induction":
            img = Image.open(slide_path).convert("RGB")
            w, h = img.size
            thumnail = np.array(img.resize((int(w / 16), int(h / 16)), Image.BILINEAR))
            gray_image = cv2.cvtColor(thumnail, cv2.COLOR_RGB2GRAY)
            threshold_val, _ = cv2.threshold(
                gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            img.close()
            for patch in patches:
                query_polygon = patch.coordinates.to_polygon()
                # 주의: annotation_filter() True면 겹침 없음
                is_overlapped = not annotation_filter(query_polygon)
                if not is_overlapped:
                    instance_labels.append(0.0)
                    continue

                gray_patch = cv2.cvtColor(patch.image_array, cv2.COLOR_RGB2GRAY)
                n_tissue = float((gray_patch <= threshold_val).sum())
                instance_labels.append(n_tissue)

            total = np.sum(instance_labels)
            if total > 0.0:
                instance_labels = np.array(instance_labels) / total
            instance_labels = torch.tensor(instance_labels, dtype=torch.float32)

        if inplace:
            self.instance_labels = instance_labels
        return instance_labels


@dataclass
class Batch:
    """key -> DataPoint 로 묶어둔 배치 컨테이너.

    Example:
        >>> from rank_induction.datasets import Batch
        >>> train_batch = Batch.from_root_path(".../feature/resnet50_3rd_20x_h5/train")
        >>> test_batch = Batch.from_root_path(".../feature/resnet50_3rd_20x_h5/test")
        >>> print(train_batch[0])
        DataPoint(
            key='tumor_001',
            h5_path='/vast/AI_team/dataset/CAMELYON16/feature/resnet50_3rd_20x_h5/train/M/tumor_001.h5',
            bag_label=tensor(1),
            instance_label=None
        )
    """

    data: Dict[str, DataPoint]

    def __post_init__(self):
        key2idx = {key: idx for idx, key in enumerate(self.data.keys())}
        idx2key = {idx: key for key, idx in key2idx.items()}
        self.key2idx = key2idx
        self.idx2key = idx2key

    def __repr__(self) -> str:
        return f"Batch(N={len(self.data)})"

    def __getitem__(self, idx: Union[int, str]) -> DataPoint:
        if idx not in self.idx2key and idx not in self.key2idx:
            raise ValueError(f"Invalid index: {idx}")

        if idx in self.idx2key:
            return self.data[self.idx2key[idx]]
        elif idx in self.key2idx:
            return self.data[self.key2idx[idx]]

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> Iterator[DataPoint]:
        return iter(self.data.values())

    @property
    def labels(self) -> torch.Tensor:
        """N개의 shape() = 스칼라 -> shape(N,)"""
        return torch.stack([data_point.bag_label for data_point in self.data.values()])

    @property
    def keys(self) -> List[str]:
        return list(self.data.keys())

    @classmethod
    def from_list(cls, data: List[DataPoint]) -> Batch:
        data = {data_point.key: data_point for data_point in data}
        return cls(data=data)

    @classmethod
    def from_root_path(cls, root_path: str) -> Batch:
        data = dict()
        for dir_path, _, filenames in os.walk(root_path):
            for filename in filenames:
                if not filename.endswith(".h5"):
                    continue

                h5_path = os.path.join(dir_path, filename)
                data_point = DataPoint.from_path(h5_path)
                label = os.path.basename(os.path.dirname(h5_path))  # "M" or "N"
                data_point.bag_label = torch.tensor(BinaryLabels[label].value)
                data[data_point.key] = data_point

        return cls(data=data)

    def add_instance_labels_from_cache(self, paths: List[str]) -> None:
        """Instance label을 캐시 numpy로부터 불러옴"""
        for path in paths:
            key = get_key(path)
            if key not in self.data:
                print(f"Key {key} not found in data")
                continue

            instance_labels = torch.from_numpy(np.load(path))
            self.data[key].instance_labels = instance_labels

        return
    
    def keep_annotation(self, fraction: float=1.0) -> None:
        """
        Annotation fraction을 유지하면서, 양성 bag의 인스턴스 라벨을 zeros로 대치
        
        Note:
            RankNetLoss의 경우, bag_true가 1이고 instance_true가 0이 아닌 경우에만 
            attention loss를 추가로 계산해줌.
            즉, bag_true 1인데, instance_true가 0인경우: Fraction of expert-annotation
            -> 이 경우는 bag_loss만을 이용해서 계산됨.
            
        """
        # 양성 bag만 대상으로 동작
        positive_bags = [
            dp for dp in self
            if int(dp.bag_label.item()) == 1
        ]
        n_pos = len(positive_bags)
        
        # 0 < fraction < 1.0: 양성 bag 중 fraction 비율만 유지
        n_keep = int(round(n_pos * fraction))
        n_keep = max(0, min(n_keep, n_pos))
        keep_indices = set(random.sample(range(n_pos), n_keep))
        
        for idx, data_point in enumerate(positive_bags):
            if idx not in keep_indices:
                data_point.instance_labels = torch.zeros_like(data_point.instance_labels)
                
        return


class MILDataset(Dataset):
    """슈퍼클레스

    Example:
        >>> from rank_induction.datasets import MILDataset
        >>> base_mil_dataset = MILDataset(batch)
        >>> train_dataset, val_dataset = base_mil_dataset.train_test_split(random_state=2025)
        >>> print(len(train_dataset), len(val_dataset))
        103 26

        >>> feature, label = train_dataset[0]
        >>> print(feature.shape, label.shape)
        >>> feature, label = val_dataset[0]
        >>> print(feature.shape, label.shape)
        torch.Size([20940, 1024]) torch.Size([])
        torch.Size([6514, 1024]) torch.Size([])
    """

    def __init__(self, batch: Batch):
        self.batch = batch

    def __len__(self) -> int:
        return len(self.batch)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            idx (int)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: features, bag_label
                1) features: (N, D)
                2) bag_label: (1, )
        """

        data_point: DataPoint = self.batch[idx]
        features: torch.Tensor = Patches.from_feature_h5(data_point.h5_path).features
        label: torch.Tensor = data_point.bag_label.float()

        return features, label

    def train_test_split(
        self, test_size: float = 0.2, random_state: int = 2025, stratify: bool = False
    ) -> Tuple[MILDataset, MILDataset]:

        if stratify:
            y_labels = self.batch.labels
        else:
            y_labels = None

        train_batch, val_batch = train_test_split(
            list(self.batch.data.values()),
            test_size=test_size,
            random_state=random_state,
            stratify=y_labels,
        )

        train_batch: List[DataPoint]
        val_batch: List[DataPoint]

        train_batch = Batch.from_list(train_batch)
        val_batch = Batch.from_list(val_batch)
        return MILDataset(train_batch), MILDataset(val_batch)


class MILCachedDataset(MILDataset):
    def __init__(self, batch: Batch):
        super().__init__(batch)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            idx (int)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: features, bag_label, instance_labels
                1) features: (N, D)
                2) bag_label: (1, )
                3) instance_labels: (N, )
        """

        data_point: DataPoint = self.batch[idx]
        features: torch.Tensor = Patches.from_feature_h5(data_point.h5_path).features
        label: torch.Tensor = data_point.bag_label.float()
        instance_labels: torch.Tensor = data_point.instance_labels

        return features, label, instance_labels

    def train_test_split(
        self, test_size: float = 0.2, random_state: int = 2025, stratify: bool = False
    ) -> Tuple[MILCachedDataset, MILCachedDataset]:
        if stratify:
            y_labels = self.batch.labels
        else:
            y_labels = None

        train_batch, val_batch = train_test_split(
            list(self.batch.data.values()),
            test_size=test_size,
            random_state=random_state,
            stratify=y_labels,
        )

        train_batch = Batch.from_list(train_batch)
        val_batch = Batch.from_list(val_batch)
        return MILCachedDataset(train_batch), MILCachedDataset(val_batch)


def get_balanced_weight_sequence(dataset: MILDataset) -> torch.Tensor:
    """Compute a balanced weight sequence for MIL dataset.

    Args:
        dataset (MILDataset): The dataset containing bag labels.

    Returns:
        torch.Tensor: A tensor of weights for each bag, balancing the class distribution.

    Example:
        >>> dataset = MILDataset(...)
        >>> weights = get_balanced_weight_sequence(dataset)
        >>> print(weights.shape)  # torch.Size([num_samples])
    """

    bag_labels: torch.Tensor = dataset.batch.labels
    n_pos = bag_labels.sum().item()
    n_total = len(bag_labels)

    weight_per_class = [n_total / (n_total - n_pos), n_total / (n_pos)]

    balanced_weight = [weight_per_class[int(bag_label)] for bag_label in bag_labels]

    return torch.tensor(balanced_weight, dtype=torch.float32)


def stratified_subsample_dataset(dataset, sampling_ratio: float):
    """
    dataset.batch.labels를 이용해서 label 비율을 유지한 채로 sampling_ratio 비율만큼 subsample
    """
    labels = dataset.batch.labels
    labels = labels.cpu().numpy()

    indices_by_label: dict[int, list[int]] = defaultdict(list)
    for idx, y in enumerate(labels):
        indices_by_label[int(y)].append(idx)

    keep_indices = []
    for y, idxs in indices_by_label.items():
        n_total = len(idxs)
        n_keep = int(round(n_total * sampling_ratio))

        chosen = random.sample(idxs, n_keep)
        keep_indices.extend(chosen)

    keep_indices.sort()

    sampled_points = [dataset.batch[i] for i in keep_indices]

    return type(dataset)(Batch.from_list(sampled_points))
