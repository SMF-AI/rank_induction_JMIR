import io

from typing import List

import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Polygon
from shapely.errors import GEOSException


class PatchFilter:
    """PatchFilter
    전달되는 filter_funcs에 하나라도 True조건을 만족하면 True로 반환

    Args:
        filter_funcs (List[callable]): 필터링 조건 함수의 목록

    Example:
        >>> patch_filter = PatchFilter([check_small_compressed_format])
        >>> patch_filter(np.asarray(patch)) // 배경 이미지
        True

    """

    def __init__(self, filter_funcs: List[callable]):
        self.filter_funcs = filter_funcs
        self.__post_init__()

    def __post_init__(self):
        if not isinstance(self.filter_funcs, list):
            raise ValueError("filter_funcs must be a list of functions")

    def __call__(self, image: np.ndarray) -> bool:
        """조건식들의 하나라도 True인 경우, True의 Early return"""
        for func in self.filter_funcs:
            if func(image):
                return True

        return False


class AnnotationFilter:
    """Annotation과 Query polygon이 얼마나 겹친지 확인하는 클레스
    - True인 경우, 겹침이 없음을 의미.
    """

    def __init__(self, label_polygons: List[Polygon], threshold: float = 0.05) -> None:
        self.label_polygons = label_polygons
        self.threshold = threshold

    def __call__(self, polygon: Polygon) -> bool:
        for label_polygon in self.label_polygons:
            try:
                intersection: Polygon = label_polygon.intersection(polygon)
            except GEOSException:
                label_polygon = label_polygon.buffer(0)
                intersection: Polygon = label_polygon.intersection(polygon)

            if intersection.is_empty:
                continue

            overlap = intersection.area / polygon.area
            if overlap > self.threshold:
                return False

        return True


def check_small_compressed_format(image: np.ndarray, min_kb=10) -> bool:
    """
    Check if the image is in the small compressed format.
    :param image_array: The image array.
    :return: True if the image is in the small compressed format, False otherwise.
    """

    image_bytes = io.BytesIO()
    Image.fromarray(image).save(image_bytes, format="JPEG")
    compressed_size = image_bytes.tell() / 1024  # size (bytes) -> kb
    if compressed_size <= min_kb:
        return True

    return False


def check_low_coverage_hsv_range(
    image: np.array,
    min_coverage: float = 0.45,
    h_range=(90, 180),
    s_range=(8, 255),
    v_range=(150, 255),
):
    """H&E 색상범위(HSV)에 해당하는 픽셀이 min_coverage을 넘는지 확인. coverage을 넘으면 False 못넘으면 True

    Parameters:
        - rgb_image (np.array): Input image in RGB format.
        - min_coverage (float): Minimum coverage threshold for valid pixels (between 0 and 1).
        - h_range (tuple): The acceptable Hue range (min, max).
        - s_range (tuple): The acceptable Saturation range (min, max).
        - v_range (tuple): The acceptable Value range (min, max).

    Returns:
        - bool: coverage을 넘으면 False 못넘으면 True (즉, True면 제외)
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # binary로 반환
    h_mask = cv2.inRange(hsv_image[:, :, 0], h_range[0], h_range[1])  # 255가 해당영역
    s_mask = cv2.inRange(hsv_image[:, :, 1], s_range[0], s_range[1])
    v_mask = cv2.inRange(hsv_image[:, :, 2], v_range[0], v_range[1])

    combined_mask = cv2.bitwise_and(h_mask, s_mask)  # 0, or 1
    combined_mask = cv2.bitwise_and(combined_mask, v_mask)

    eligible_pixels = np.sum(combined_mask > 0)
    total_pixels = image.shape[0] * image.shape[1]

    coverage = eligible_pixels / total_pixels
    return coverage <= min_coverage
