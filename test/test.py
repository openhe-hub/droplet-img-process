import json
from math import factorial

import cv2
import numpy as np

from src.process import dilate_edges, close_edges

id = 1
raw_path = f'../assets/case{id}/raw_input.jpg'
path = f'../assets/case{id}/input.jpg'
background_path = r'../assets/background.jpg'
dataset_path = f'../assets/case{id}/dataset.json'
tolerance = 35

def remove_color_from_image(image, coord, radius, tolerance=30):
    bs, gs, rs = [], [], []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            dist = ((i-coord[1])**2 + (j-coord[0])**2)**0.5
            if radius * 0.4 <= dist <= radius * 0.7:
                bs.append(image[i, j][0])
                gs.append(image[i, j][1])
                rs.append(image[i, j][2])
    bs, gs, rs = np.array(bs), np.array(gs), np.array(rs)
    b_mean, g_mean, r_mean = bs.mean(), gs.mean(), rs.mean()
    color_bgr = [b_mean, g_mean, r_mean]

    color_hsv = cv2.cvtColor(np.uint8([[color_bgr]]), cv2.COLOR_BGR2HSV)[0][0]

    lower_bound = np.array([max(0, color_hsv[0] - tolerance), max(0, color_hsv[1] - tolerance), max(0, color_hsv[2] - tolerance)])
    upper_bound = np.array([min(255, color_hsv[0] + tolerance), min(255, color_hsv[1] + tolerance), min(255, color_hsv[2] + tolerance)])

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    color_mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    adjusted_image = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(color_mask))
    return adjusted_image

def find_contours(img, render_img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 40, 255, cv2.THRESH_BINARY)
    cv2.imshow('binary', binary_image)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    binary_image = cv2.GaussianBlur(binary_image, (5, 5), sigmaX=1.0)
    cv2.imshow('binary-2', binary_image)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # 查找圆度最高的轮廓
    filtered_contour = []
    highest_circularity = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter < 500 or area < 500 or area > 80000:  # 避免除以零
            continue
        print(area)
        filtered_contour.append(contour)
        # circularity = 4 * np.pi * (area / (perimeter * perimeter))  # 计算圆度
        # # print(circularity)
        # if circularity > highest_circularity:
        #     highest_circularity = circularity
        #     filtered_contour = [contour]

    print(len(contours))
    cv2.drawContours(render_img, filtered_contour, -1, (0, 255, 255), 2)  # 绿色线条
    return render_img

def find_finger(coords, center, radius, render_img, dist_thres = 2):
    finger_cnt = 0
    if_at_finger = False
    for point in coords:
        x, y = point[0], point[1]
        distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        if distance > radius + dist_thres:
            cv2.circle(render_img, (x, y), 3, (0, 0, 255), -1)
            if not if_at_finger:
                finger_cnt += 1
                if_at_finger = True
        elif distance <= radius + dist_thres:
            if if_at_finger:
                if_at_finger = False
    cv2.putText(render_img, f'FINGER_NUM = {finger_cnt}', (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return render_img

if __name__ == '__main__':
    background = cv2.imread(background_path)
    raw_img = cv2.imread(raw_path)
    input = cv2.imread(path)
    dataset = json.load(open(dataset_path))
    diff1 = cv2.absdiff(raw_img, background)
    cv2.imshow('background', background)
    cv2.imshow('raw', raw_img)
    cv2.imshow('detect', input)
    cv2.imshow('diff1', diff1)
    diff2 = remove_color_from_image(diff1, dataset['circle_center'], dataset['circle_radius'], tolerance=tolerance)
    cv2.imshow('diff2', diff2)
    result = find_contours(diff2, input)
    cv2.imshow('result', result)
    finger_result = find_finger(dataset['contour'], dataset['circle_center'], dataset['circle_radius'], result)
    cv2.imshow('finger_result', finger_result)
    cv2.waitKey(0)

