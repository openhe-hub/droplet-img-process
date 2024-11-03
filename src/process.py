import csv
from typing import Sequence
import cv2
import numpy as np

# params
CANNY_THRESHOLD1 = 100.0
CANNY_THRESHOLD2 = 200.0
MIN_CONTOUR_AREA = 1500.0
MIN_CONTOUR_ARC_LEN = 80.0
MIN_CONTOUR_CIRCULARITY = 0.5
MAX_CONTOUR_CIRCULARITY = 1.5
DILATE_KERNEL_SIZE = (3,3)
DILATE_ITERATION_NUM = 2
DENOISE_KERNEL_SIZE = (7,7)
DENOISE_SIGMA_X = 1.0
CLOSE_KERNEL_SIZE = (3,3)
CLOSE_ITERATION_NUM = 1
BINARY_THRESHOLD = 50

def load_img(path: str) -> cv2.Mat:
    try:
        img = cv2.imread(path)
        return img
    except Exception as e:
        print(f"Err occurs while reading {path}: {e}")

def diff_img(img1: cv2.Mat, img2: cv2.Mat) -> cv2.Mat:
    diff = cv2.absdiff(img1, img2)
    return diff

def gbr_to_gray(origin_img: cv2.Mat) -> cv2.Mat:
    return cv2.cvtColor(origin_img, cv2.COLOR_BGR2GRAY)

def gray_to_gbr(origin_img: cv2.Mat) -> cv2.Mat:
    return cv2.cvtColor(origin_img, cv2.COLOR_GRAY2BGR)

def gbr_to_binary(origin_img: cv2.Mat) -> cv2.Mat:
    _, binary_img = cv2.threshold(origin_img, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)
    return binary_img

def find_contour_by_canny(origin_img: cv2.Mat) -> Sequence[cv2.Mat]:
    edges = cv2.Canny(origin_img, CANNY_THRESHOLD1, CANNY_THRESHOLD2)
    # edges = dilate_edges(edges)
    # edges = close_edges(edges)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print(f'contour coordinates: {contours}')
    # print(f'hierarchy map: {hierarchy}')
    return contours

def filter_contour(contours: Sequence[cv2.Mat]) -> Sequence[cv2.Mat]:
    filtered_contours = []
    areas = [cv2.contourArea(contour) for contour in contours]
    areas.sort(reverse=True)
    for contour in contours:
        # 1. filter by area & circumstance
        area = cv2.contourArea(contour)
        length = cv2.arcLength(contour, closed=True)
        if area == areas[0]:
            filtered_contours.append(contour)
            break
        # arc_length = cv2.arcLength(contour, True)
        # if area >= MIN_CONTOUR_AREA and arc_length >= MIN_CONTOUR_ARC_LEN:
        #     # 2. filter by circle-like shape
        #     circularity = 4 * np.pi * (area / np.power(arc_length, 2))
        #     if not (MIN_CONTOUR_CIRCULARITY <= circularity <= MAX_CONTOUR_CIRCULARITY):
        #         filtered_contours.append(contour)
    # print(f"origin contour num = {len(contours)}, filtered num = {len(filtered_contours)}")
    return filtered_contours

def save_contour(contour: cv2.Mat, path: str):
    with open(path, mode='w', newline='') as file:
        writer = csv.writer(file)
        for point in contour:
            x, y = point[0]
            writer.writerow([x, y])

def denoise_img(origin_img: cv2.Mat) -> cv2.Mat:
    return cv2.GaussianBlur(origin_img, DENOISE_KERNEL_SIZE, DENOISE_SIGMA_X)

def dilate_edges(edges: cv2.Mat) -> cv2.Mat:
    kernel = np.ones(DILATE_KERNEL_SIZE, np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=DILATE_ITERATION_NUM)
    return dilated_edges

def close_edges(edges: cv2.Mat) -> cv2.Mat:
    kernel = np.ones(CLOSE_KERNEL_SIZE, np.uint8)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=CLOSE_ITERATION_NUM)
    return closed_edges

