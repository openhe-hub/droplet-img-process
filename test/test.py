import json
from math import factorial

import cv2
import numpy as np

id = 1
raw_path = f'../assets/case{id}/raw_input.jpg'
path = f'../assets/case{id}/input.jpg'
background_path = r'../assets/background.jpg'
dataset_path = f'../assets/case{id}/dataset.json'
tolerance = 30

def remove_color_from_image(image, coord, tolerance=30):
    color_bgr = image[coord[1], coord[0]]
    # 转换到HSV色彩空间
    color_hsv = cv2.cvtColor(np.uint8([[color_bgr]]), cv2.COLOR_BGR2HSV)[0][0]

    # 设置HSV范围，允许一定容忍度
    lower_bound = np.array([max(0, color_hsv[0] - tolerance), max(0, color_hsv[1] - tolerance), max(0, color_hsv[2] - tolerance)])
    upper_bound = np.array([min(179, color_hsv[0] + tolerance), min(255, color_hsv[1] + tolerance), min(255, color_hsv[2] + tolerance)])

    # 转换整个图像到HSV空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 创建掩码并去除指定颜色范围
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    adjusted_image = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))
    return adjusted_image

def find_contours(img, render_img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 查找圆度最高的轮廓
    most_circular_contour = None
    highest_circularity = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter < 1000 or area < 500:  # 避免除以零
            continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))  # 计算圆度
        # print(circularity)
        if circularity > highest_circularity:
            highest_circularity = circularity
            most_circular_contour = contour

    cv2.drawContours(render_img, [most_circular_contour], -1, (0, 255, 255), 2)  # 绿色线条
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
    diff2 = remove_color_from_image(diff1, dataset['circle_center'], tolerance=tolerance)
    cv2.imshow('diff2', diff2)
    result = find_contours(diff2, input)
    cv2.imshow('result', result)
    finger_result = find_finger(dataset['contour'], dataset['circle_center'], dataset['circle_radius'], result)
    cv2.imshow('finger_result', finger_result)
    cv2.waitKey(0)

