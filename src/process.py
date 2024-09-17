from typing import Sequence
import cv2
import numpy as np

# params
CANNY_THRESHOLD1 = 100.0
CANNY_THRESHOLD2 = 200.0
MIN_CONTOUR_AREA = 200.0
MIN_CONTOUR_ARC_LEN = 100.0
MIN_CONTOUR_CIRCULARITY = 0.5
MAX_CONTOUR_CIRCULARITY = 1.5
DILATE_KERNEL_SIZE = (5,5)
DILATE_ITERATION_NUM = 1
DENOISE_KERNEL_SIZE = (5,5)
DENOISE_SIGMA_X = 0.001

def load_img(path: str) -> cv2.Mat:
    try:
        img = cv2.imread(path)
        return img
    except Exception as e:
        print(f"Err occurs while reading {path}: {e}")

def gbr_to_gray(origin_img: cv2.Mat) -> cv2.Mat:
    return cv2.cvtColor(origin_img, cv2.COLOR_BGR2GRAY)

def gray_to_gbr(origin_img: cv2.Mat) -> cv2.Mat:
    return cv2.cvtColor(origin_img, cv2.COLOR_GRAY2BGR)

def find_contour_by_canny(origin_img: cv2.Mat) -> Sequence[cv2.Mat]:
    edges = cv2.Canny(origin_img, CANNY_THRESHOLD1, CANNY_THRESHOLD2)
    edges = dilate_edges(edges)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(f'contour coordinates: {contours}')
    print(f'hierarchy map: {hierarchy}')
    return contours

def filter_contour(contours: Sequence[cv2.Mat]) -> Sequence[cv2.Mat]:
    filtered_contours = []
    for contour in contours:
        # 1. filter by area & circumstance
        area = cv2.contourArea(contour)
        arc_length = cv2.arcLength(contour, True)
        if area >= MIN_CONTOUR_AREA and arc_length >= MIN_CONTOUR_ARC_LEN:
            # filtered_contours.append(contour)
            # continue
            # 2. filter by circle-like shape
            circularity = 4 * np.pi * (area / np.power(arc_length, 2))
            if not (MIN_CONTOUR_CIRCULARITY <= circularity <= MAX_CONTOUR_CIRCULARITY):
                filtered_contours.append(contour)
    print(f"origin contour num = {len(contours)}, filtered num = {len(filtered_contours)}")
    return filtered_contours

def denoise_img(origin_img: cv2.Mat) -> cv2.Mat:
    return cv2.GaussianBlur(origin_img, DENOISE_KERNEL_SIZE, DENOISE_SIGMA_X)

def dilate_edges(edges: cv2.Mat) -> cv2.Mat:
    kernel = np.ones(DILATE_KERNEL_SIZE, np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=DILATE_ITERATION_NUM)
    return dilated_edges

