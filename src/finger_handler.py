import json
import math
import cv2
import numpy as np

from process import dilate_edges, close_edges
from droplet import Droplet

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
            if ((i-coord[1])**2 + (j-coord[0])**2)**0.5 < radius * 0.95:
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
    _, binary_image = cv2.threshold(gray_image, 30, 255, cv2.THRESH_BINARY)
    # cv2.imshow('binary', binary_image)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    binary_image = cv2.GaussianBlur(binary_image, (5, 5), sigmaX=1.0)
    # cv2.imshow('binary-2', binary_image)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # 查找圆度最高的轮廓
    filtered_contour = []
    highest_circularity = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter < 500 or area < 20000 or area > 80000:  # 避免除以零
            continue
        # print(area)
        filtered_contour.append(contour)
        # circularity = 4 * np.pi * (area / (perimeter * perimeter))  # 计算圆度
        # # print(circularity)
        # if circularity > highest_circularity:
        #     highest_circularity = circularity
        #     filtered_contour = [contour]

    # print(len(contours))
    cv2.drawContours(render_img, filtered_contour, -1, (0, 255, 255), 2)  # 绿色线条

    # center = circle_center
    # center_x, center_y = center
    # max_dist, min_dist = -np.inf, np.inf
    # for pt in filtered_contour:
    #     x, y = pt
    #     dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    #     max_dist = max(max_dist, dist)
    #     min_dist = min(min_dist, dist)
    #
    # cv2.circle(render_img, center, int(max_dist), [0, 255, 0], thickness=2)

    return render_img

def calc_tang(x1, y1, x2, y2, x0, y0):
    _x1 = x1 - x0
    _x2 = x2 - x0
    _y1 = y1 - y0
    _y2 = y2 - y0
    r1 = math.hypot(_x1, _y1)
    theta1 = math.atan2(_y1, _x1)
    r2 = math.hypot(_x2, _y2)
    theta2 = math.atan2(_y2, _x2)
    delta_r = r2 - r1
    delta_theta = theta2 - theta1
    if delta_theta == 0: return 10000
    derivative = min(abs(delta_r / delta_theta), 10000)
    return derivative

def find_finger(pts, center, radius, img):
    dist_map = {}
    tang_map = {}
    center_x, center_y = center
    prev_pt = (pts[-1][0], pts[-1][1])
    for pt in pts:
        x, y = pt
        dist = (x - center_x) ** 2 + (y - center_y) ** 2
        dist_map[(x, y)] = dist
        tang_map[(x, y)] = calc_tang(x, y, prev_pt[0], prev_pt[1], center_x, center_y)
        prev_pt = (x, y)

    dist_arr = list(dist_map.values())
    tang_arr = list(tang_map.values())
    r_avg = np.percentile(dist_arr, 55)
    r_min = np.percentile(dist_arr, 30)
    tang_avg = np.percentile(tang_arr, 50)
    tang_min = np.percentile(tang_arr, 30)

    # filter
    filtered_pts = []
    for k, v in tang_map.items():
        if ((v > tang_avg and dist_map[k] > r_min) or
                (dist_map[k] > r_avg)):
            filtered_pts.append(k)

    # group
    groups = []
    curr_group = []
    thres = 80
    for idx, pt in enumerate(filtered_pts):
        if idx == len(filtered_pts) - 1: continue
        next_pt = filtered_pts[idx + 1]
        dist = (pt[0]-next_pt[0]) ** 2 + (pt[1]-next_pt[1]) ** 2
        if dist <= thres:
            curr_group.append(pt)
        else:
            if len(curr_group) > 0:
                if len(curr_group) >= 3:
                    groups.append(curr_group)
                curr_group = []

    for group in groups:
        for idx, pt in enumerate(group):
            if idx == len(group) - 1: continue
            cv2.line(img, pt, group[idx+1], [0,0,255], thickness=3)

    return len(groups), img

def handle_finger(droplet: Droplet, raw_img: cv2.Mat, input: cv2.Mat, background: cv2.Mat) -> [Droplet, cv2.Mat]:
    # diff1 = cv2.absdiff(raw_img, background)
    # diff2 = remove_color_from_image(diff1, droplet.circle_center, droplet.circle_radius, tolerance=tolerance)
    # result = find_contours(diff2, input)
    finger_cnt, finger_img = find_finger(droplet.contour, droplet.circle_center, droplet.circle_radius, input)
    droplet.finger_num = finger_cnt
    return droplet, finger_img

# if __name__ == '__main__':
#     background = cv2.imread(background_path)
#     raw_img = cv2.imread(raw_path)
#     input = cv2.imread(path)
#     dataset = json.load(open(dataset_path))
#     diff1 = cv2.absdiff(raw_img, background)
#     cv2.imshow('background', background)
#     cv2.imshow('raw', raw_img)
#     cv2.imshow('detect', input)
#     cv2.imshow('diff1', diff1)
#     diff2 = remove_color_from_image(diff1, dataset['circle_center'], dataset['circle_radius'], tolerance=tolerance)
#     cv2.imshow('diff2', diff2)
#     result = find_contours(diff2, input)
#     cv2.imshow('result', result)
#     finger_result = find_finger(dataset['contour'], dataset['circle_center'], dataset['circle_radius'], result)
#     cv2.imshow('finger_result', finger_result)
#     cv2.waitKey(0)

