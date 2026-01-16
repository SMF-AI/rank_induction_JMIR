from __future__ import annotations

import os
import queue
import glob
import numbers
import multiprocessing
from enum import Enum
from logging import Logger
from typing import List, Iterable, Tuple, Set, Callable, Literal
from dataclasses import dataclass, field
from multiprocessing import Process, Queue

import cv2
import torch
import h5py
import shapely.errors
import tqdm
import shapely
import openslide
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from skimage.filters import threshold_multiotsu
from PIL import Image, ImageDraw, ImageOps
from shapely.geometry import Polygon
from openslide import open_slide, OpenSlide
from openslide.deepzoom import DeepZoomGenerator

from rank_induction.misc import read_json

Image.MAX_IMAGE_PIXELS = None


@dataclass
class Coordinates:
    x_min: int = None
    y_min: int = None
    x_max: int = None
    y_max: int = None

    def __repr__(self) -> str:
        return f"Coordinates(x_min={self.x_min}, y_min={self.y_min}, x_max={self.x_max}, y_max={self.y_max})"

    def to_string(self) -> str:
        return f"{self.x_min}_{self.y_min}_{self.x_max}_{self.y_max}"

    def to_polygon(self) -> Polygon:
        return Polygon(
            [
                (self.x_min, self.y_min),
                (self.x_max, self.y_min),
                (self.x_max, self.y_max),
                (self.x_min, self.y_max),
            ]
        )

    def to_coco(self, size: tuple) -> List[float, float, float, float]:
        w = self.x_max - self.x_min
        h = self.y_max - self.y_min
        return [
            self.x_min / size[0],
            self.y_min / size[1],
            w / size[0],
            h / size[1],
        ]

    def to_list(self) -> List[float, float, float, float]:
        return [self.x_min, self.y_min, self.x_max, self.y_max]


@dataclass
class Polygons:
    path: str = str()
    data: List[Polygon] = field(default_factory=list)

    """Polygons
    - polygons의 집합(=Annotation set)
    
    Attributes
        - path: annotation path
        - data: annotation polygons
    """

    def __repr__(self) -> str:
        n_polygon: int = len(self.data) if self.data else 0
        return f"QuPathPloygon(path={self.path}, N polygons={n_polygon})"


class BinaryLabels(Enum):
    """이진분류 라벨 상수 표현의 열거형

    Example:
        # class index를 이용하여 라벨 추출
        >>> from seedp.data_models import BinaryLabels
        >>> y_pred = torch.tensor([[0.1, 0.9]])
        >>> class_idx = torch.argmax(y_pred).item()
        >>> print(BinaryLabels(class_idx).name)
        'M'

        # class name을 이용하여, 인덱스 추출
        >>> slide_label:str = os.path.basename(slide_path)
        >>> print(slide_label)
        'N'
        >>> BinaryLabels[slide_label]
        <BinaryLabels.N: 0>
        >>> BinaryLabels[slide_label].value
        0


    """

    N = 0
    M = 1


# TODO
@dataclass
class Patch:
    """패치의 데이터클레스

    Args:
        address (tuple): col, row
        coordinates (Coordinates): Slide level 0에서의 x1, y1, x2, y2

    Example:
        # Array로부터 패치 생성
        >>> image_array:np.ndarray = ....
        >>> patch = Patch(image_array)
        >>> patch
        Patch(image_shape=(512, 512, 3), path=None, label=Labels.unknown, coordinates=None, level=None)

        # 파일로부터 생성
        >>> patch = Patch.from_file("test_path.png", label=Labels.unknown)
        >>> patch
        Patch(image_shape=(512, 512, 3), path=None, label=Labels.unknown, coordinates=None, level=None)
        >>> patch.close() # 메모리 해제


        # 좌표와 함께 생성
        >>> patch = Patch(label=Labels.unknown, coordinates=Coordinates(100, 105, 200, 200))
        >>> patch.coordinates.x_min
        100
        >>> patch.coordinates.y_min
        105
        >>> patch.coordinates.x_max
        200

    """

    image_array: np.ndarray = None
    confidences: np.ndarray = None
    coordinates: Coordinates = None
    feature: torch.Tensor = None
    address: Tuple[int, int] = None
    label: str = None

    slide_name: str = None
    path: str = None
    level: int = None

    @classmethod
    def from_file(cls, path: str, **kwargs: dict) -> Patch:
        image_array = np.array(Image.open(path))
        return Patch(image_array, **kwargs)

    def __repr__(self) -> str:
        image_shape = (
            "None" if self.image_array is None else str(self.image_array.shape)
        )
        return (
            "Patch("
            f"image_shape={image_shape}, path={self.path}, label={self.label}, "
            f"coordinates={str(self.coordinates)}, address={self.address}, "
            f"confidences={str(self.confidences)}"
            ")"
        )

    def load(self):
        if not self.path:
            raise ValueError("path is None")

        self.image_array = np.array(Image.open(self.path))

        return

    def close(self):
        del self.image_array

        return


# TODO
@dataclass
class Patches:
    """패치의 복수의 집합"""

    data: List[Patch] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    dimension: tuple = None

    def __getitem__(self, i: int) -> Patch:
        return self.data[i]

    def __iter__(self):
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self):
        return f"Patches(N={len(self.data)})"

    def build_feature_cube_pad(self, size: tuple) -> torch.Tensor:
        """패딩이 필요한 feature cube 생성(패딩영역은 마지막 채널)

        Args:
            size (tuple): size (rows, cols)

        Returns:
            torch.Tensor: feature cube (1, 3, row, col)
        """
        feature_cube = torch.zeros((1, 3, *size), dtype=torch.float32)
        for patch in self.data:
            col, row = patch.address
            if patch.confidences is None:
                confidences = torch.tensor([0, 0, 0], dtype=torch.float32)
                feature_cube[0, :, row, col] = confidences
                continue

            feature_cube[0, :, row, col] = torch.concat(
                [
                    torch.from_numpy(patch.confidences).float(),
                    torch.tensor([0], dtype=torch.float32),
                ],
            )

        return feature_cube

    def build_feature_cube(self, size: tuple) -> torch.Tensor:
        """패딩이 필요없는 feature cube 생성(패딩영역은 마지막 채널)

        Args:
            size (tuple): size (rows, cols)

        Returns:
            np.ndarray: feature cube
        """
        feature_cube = torch.zeros((1, 3, *size), dtype=torch.float32)
        for patch in self.data:
            col, row = patch.address
            if patch.confidences is None:
                confidences = torch.tensor([0, 0, 0], dtype=torch.float32)
                feature_cube[0, :, row, col] = confidences
                continue

            feature_cube[0, :, row, col] = torch.from_numpy(patch.confidences).float()

        return feature_cube

    @classmethod
    def from_queue(cls, queue: queue.Queue, drop_empty_patch: bool = True) -> Patches:
        """병렬처리시 Queue로부터 patches을 생성함

        Args:
            queue (Queue): 병렬처리할 때 사용된 shared memory queue
            drop_empty_patch (bool): 패치이미지중 필터된 이미지의 삭제 여부
                (True: 해당 패치 데이터클레스 삭제)

        Returns:
            patches (Patches): 패치의 복수의 집합

        Example:

            >>> tiler = Tiler(
                    config.tile_size,
                    config.overlap,
                    config.limit_bounds,
                    deepzoom_level=deepzoom_level,
                    n_workers=config.tile_workers,
                    patch_filter=patch_filter,
                    logger=logger,
                )
            >>> output_queue: queue.Queue = tiler.do_tile(query.slide_path)
            >>> tiler.join() # 병렬처리 종료대기
            >>> patches = Patches.from_queue(output_queue)

        """
        data = list()
        while not queue.empty():
            patch = queue.get()

            if not drop_empty_patch:
                data.append(patch)
                continue

            if patch.image_array is not None:
                data.append(patch)

        return Patches(data)

    def save(self, dir: str, format: Literal["png", "h5"]) -> None:
        """패치들을 저장

        Example:
            >>> # PNG로 저장
            >>> patches.save("slide_name", format="png")
            >>> os.listdir("slide_name")
            [
                "slide_name_15_23.png",
                "slide_name_15_24.png",
                ...
            ]

            >>> # h5로 저장
            >>> patches.save("slide_name.h5", format="h5")

            >>> # h5로부터 로드
            >>> from seedp.data_models import Patches
            >>> patches = Patches.from_h5("slide_name.h5")

        Args:
            dir (str): 목적지 경로
            format (Literal["png", "h5"], optional): 저장포맷. Defaults to "h5".
        """
        if format == "png":
            os.makedirs(dir, exist_ok=True)
            for patch in self.data:
                Image.fromarray(patch.image_array).save(
                    os.path.join(
                        dir,
                        f"{patch.slide_name}_{patch.address[0]}_{patch.address[1]}.png",
                    )
                )

        elif format == "h5":
            with h5py.File(dir, "w") as fh:
                for patch in self.data:
                    col, row = patch.address
                    key = f"{patch.address[0]}_{patch.address[1]}"
                    dataset = fh.create_dataset(
                        key, data=patch.image_array, compression="gzip"
                    )
                    dataset.attrs["col"] = col
                    dataset.attrs["row"] = row
                    dataset.attrs["x_min"] = patch.coordinates.x_min
                    dataset.attrs["y_min"] = patch.coordinates.y_min
                    dataset.attrs["x_max"] = patch.coordinates.x_max
                    dataset.attrs["y_max"] = patch.coordinates.y_max

        return

    def save_grid(
        self, path: str, subpatch_size: int = 256, region_size: int = 4096
    ) -> None:
        """4096x4096 패치를 16x16x256x256 그리드로 저장 (각 키는 'col_row')

        Args:
            path (str): 출력 H5 경로
            subpatch_size (int, optional): 그리드 서브패치 크기. Defaults to 256.
            region_size (int, optional): 입력 패치 크기. Defaults to 4096.
        """
        n_grid = region_size // subpatch_size

        with h5py.File(path, "w") as fh:
            for patch in self.data:
                img_t = torch.from_numpy(patch.image_array)  # (H, W, 3) uint8
                H, W = img_t.shape[:2]
                if (H, W) != (region_size, region_size):
                    # HWC -> CHW -> resize -> HWC
                    img_t = img_t.permute(2, 0, 1).float()
                    img_t = torch.nn.functional.interpolate(
                        img_t.unsqueeze(0),
                        size=(region_size, region_size),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                    img_t = img_t.permute(1, 2, 0).byte()

                img = img_t.cpu().numpy()  # (region, region, 3)

                # Slice into grid (n_grid, n_grid, subpatch_size, subpatch_size, 3)
                grid = np.empty(
                    (n_grid, n_grid, subpatch_size, subpatch_size, 3), dtype=img.dtype
                )
                for r in range(n_grid):
                    for c in range(n_grid):
                        grid[r, c] = img[
                            r * subpatch_size : (r + 1) * subpatch_size,
                            c * subpatch_size : (c + 1) * subpatch_size,
                            :,
                        ]

                col, row = patch.address
                key = f"{col}_{row}"
                dset = fh.create_dataset(key, data=grid, compression="gzip")
                dset.attrs["col"] = col
                dset.attrs["row"] = row
                dset.attrs["x_min"] = patch.coordinates.x_min
                dset.attrs["y_min"] = patch.coordinates.y_min
                dset.attrs["x_max"] = patch.coordinates.x_max
                dset.attrs["y_max"] = patch.coordinates.y_max

        return

    def close(self) -> None:
        for patch in self.data:
            del patch.image_array
        return

    @classmethod
    def from_patch_h5(cls, path: str) -> Patches:
        """HDF5 파일에서 Patches 객체를 생성

        Args:
            path (str): HDF5 파일 경로

        Returns:
            Patches: 복원된 Patches 객체

        Example:
            >>> from seedp.data_models import Patches
            >>> patches = Patches.from_patch_h5("***.h5")
            Patches(length=585)
        """
        data: List[Patch] = []

        with h5py.File(path, "r") as fh:
            for key in fh.keys():
                dataset = fh[key]

                # 이미지 배열을 불러오기
                image_array = np.array(dataset)

                # address (col, row) 복원
                col = dataset.attrs.get("col")
                row = dataset.attrs.get("row")
                address = (col, row)

                # coordinates 복원
                coordinates = Coordinates(
                    x_min=dataset.attrs.get("x_min"),
                    y_min=dataset.attrs.get("y_min"),
                    x_max=dataset.attrs.get("x_max"),
                    y_max=dataset.attrs.get("y_max"),
                )
                patch = Patch(
                    image_array=image_array, address=address, coordinates=coordinates
                )
                data.append(patch)

        return cls(data=data)

    @property
    def features(self) -> torch.Tensor:
        """
        Example:
        >>> patches = Patches.from_feature_h5("here.feature.h5")
        >>> print(patches.features.shape)
        torch.Size([585, 768])
        """
        return torch.stack([patch.feature for patch in self.data], dim=0)

    @classmethod
    def from_feature_h5(cls, path: str) -> Patches:
        """HDF5 파일에서 Patches 객체를 생성

        Args:
            path (str): HDF5 파일 경로

        Returns:
            Patches: 복원된 Patches 객체
        """
        data: List[Patch] = []

        with h5py.File(path, "r") as fh:
            for key in fh.keys():
                dataset = fh[key]

                # 이미지 배열을 불러오기
                feature = torch.from_numpy(np.array(dataset))

                # address (col, row) 복원
                col = dataset.attrs.get("col")
                row = dataset.attrs.get("row")
                address = (col, row)

                # coordinates 복원
                coordinates = Coordinates(
                    x_min=dataset.attrs.get("x_min"),
                    y_min=dataset.attrs.get("y_min"),
                    x_max=dataset.attrs.get("x_max"),
                    y_max=dataset.attrs.get("y_max"),
                )
                patch = Patch(
                    image_array=None,
                    feature=feature,
                    address=address,
                    coordinates=coordinates,
                )
                data.append(patch)

        return cls(data=data)
