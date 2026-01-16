import os
import sys
import glob
import random
import tqdm
import colorsys
from collections import defaultdict
from typing import Tuple, Dict
from functools import partial

import cv2
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms import (
    Compose,
    Resize,
    Normalize,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    ToTensor,
)
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset

from seestaina.augmentation import Augmentor
from seestaina.structure_preversing import Augmentor as StructureAugmentor

RANK_INDUCTION_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(RANK_INDUCTION_DIR)
EXP_DIR = os.path.join(ROOT_DIR, "experiments")
RANDSTAINNA_PATH = "/home/heon/dev/RandStainNA"

data_transforms = {
    "train": Compose(
        [
            Resize((224, 224)),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            ToTensor(),
        ]
    ),
    "test": Compose(
        [
            Resize((224, 224)),
            ToTensor(),
        ]
    ),
}


 


def get_dist_params(image_dir: str) -> dict:

    l_mean = list()
    l_sd = list()
    a_mean = list()
    a_sd = list()
    b_mean = list()
    b_sd = list()

    for class_dir in os.listdir(image_dir):
        image_paths = glob.glob(os.path.join(image_dir, class_dir, "*.png"))
        for filename in image_paths:
            rgb_image = cv2.imread(filename)
            lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
            L_channel, A_channel, B_channel = cv2.split(lab_image)
            l_mean.append(L_channel.mean())
            a_mean.append(A_channel.mean())
            b_mean.append(B_channel.mean())

            l_sd.append(L_channel.std())
            a_sd.append(A_channel.std())
            b_sd.append(B_channel.std())

    params = {
        "L": {
            "avg": {
                "mean": float(np.mean(l_mean)),
                "std": float(np.std(l_mean)),
            },
            "std": {
                "mean": float(np.mean(l_sd)),
                "std": float(np.std(l_sd)),
            },
        },
        "A": {
            "avg": {
                "mean": float(np.mean(a_mean)),
                "std": float(np.std(a_mean)),
            },
            "std": {
                "mean": float(np.mean(a_sd)),
                "std": float(np.std(a_sd)),
            },
        },
        "B": {
            "avg": {
                "mean": float(np.mean(b_mean)),
                "std": float(np.std(b_mean)),
            },
            "std": {
                "mean": float(np.mean(b_sd)),
                "std": float(np.std(b_sd)),
            },
        },
    }
    return params


class RandStainNAPIL:
    def __init__(self, template_path: str, randstainna_path: str = None):
        self.template_path = template_path
        self.randstainna_path = randstainna_path

    def __call__(self, image_array: Image.Image) -> Image.Image:
        image_array = self.randstainna_fn(image_array)
        image_array = cv2.cvtColor(np.array(image_array), cv2.COLOR_BGR2RGB)
        return Image.fromarray(image_array)

    def fit(self, subset_folder: Subset):
        if self.randstainna_path:
            sys.path.append(self.randstainna_path)
        else:
            sys.path.append(RANDSTAINNA_PATH)
        from randstainna import RandStainNA

        dist_params = get_dist_params(subset_folder)
        config_template = OmegaConf.load(self.template_path)
        config_template.update(dist_params)
        OmegaConf.save(config_template, self.template_path)

        self.randstainna_fn: callable = RandStainNA(
            yaml_file=self.template_path,
            std_hyper=-0.3,
            probability=0.5,
            distribution="normal",
            is_train=True,
        )

        return


class StainSepPIL:
    def __init__(
        self,
        aug_saturation: bool = True,
        aug_density: bool = True,
    ):
        self.aug_saturation = aug_saturation
        self.aug_density = aug_density
        self.distribution = "normal"

    def __call__(self, image: Image.Image) -> Image.Image:
        return self.augmentor.image_augmentation_with_stain_vector(
            image,
            aug_saturation=self.aug_saturation,
            aug_density=self.aug_density,
        )

    def fit(self, stain_cache_dir):
        self.augmentor = StructureAugmentor(dist=self.distribution, od_threshold=0.01)
        self.augmentor.load_stain_cache(stain_cache_dir)

        return


class MixAugPIL:
    def __init__(
        self,
        template_path: str,
        aug_saturation: bool = True,
        aug_density: bool = True,
        distribution="normal",
    ):
        self.template_path = template_path
        self.aug_saturation = aug_saturation
        self.aug_density = aug_density
        self.distribution = distribution

    def __call__(self, image: Image.Image) -> Image.Image:
        p = random.random()
        if p >= 0.5:
            return image

        if p >= 0.25:
            return self.randstainna_fn(image)

        return self.stain_sep(image)

    def fit(self, subset_folder, stain_cache_dir):
        self.stain_sep = StainSepPIL(self.aug_saturation, self.aug_density)
        self.stain_sep.fit(stain_cache_dir)

        self.randstainna_fn = RandStainNAPIL(self.template_path)
        self.randstainna_fn.fit(subset_folder)
        return
 


def get_dist_params(image_folder) -> dict:

    l_mean = list()
    l_sd = list()
    a_mean = list()
    a_sd = list()
    b_mean = list()
    b_sd = list()

    for image, label in image_folder:
        lab_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2LAB)
        L_channel, A_channel, B_channel = cv2.split(lab_image)
        l_mean.append(L_channel.mean())
        a_mean.append(A_channel.mean())
        b_mean.append(B_channel.mean())

        l_sd.append(L_channel.std())
        a_sd.append(A_channel.std())
        b_sd.append(B_channel.std())

    params = {
        "L": {
            "avg": {
                "mean": float(np.mean(l_mean)),
                "std": float(np.std(l_mean)),
            },
            "std": {
                "mean": float(np.mean(l_sd)),
                "std": float(np.std(l_sd)),
            },
        },
        "A": {
            "avg": {
                "mean": float(np.mean(a_mean)),
                "std": float(np.std(a_mean)),
            },
            "std": {
                "mean": float(np.mean(a_sd)),
                "std": float(np.std(a_sd)),
            },
        },
        "B": {
            "avg": {
                "mean": float(np.mean(b_mean)),
                "std": float(np.std(b_mean)),
            },
            "std": {
                "mean": float(np.mean(b_sd)),
                "std": float(np.std(b_sd)),
            },
        },
    }
    return params
 
