import numpy as np
import matplotlib.pyplot as plt
import math
import json
import cv2

dataset_path = '../assets/case1/dataset.json'
raw_img_path = '../assets/case1/raw_input.jpg'
img_path = '../assets/case1/input.jpg'


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


def stat_radius(pts, center, radius, img):
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
    tang_avg = np.percentile(tang_arr, 60)
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
        print(dist)
        if dist <= thres:
            curr_group.append(pt)
        else:
            if len(curr_group) > 0:
                groups.append(curr_group)
                curr_group = []

    for group in groups:
        for idx, pt in enumerate(group):
            if idx == len(group) - 1: continue
            cv2.line(img, pt, group[idx+1], [0,0,255], thickness=3)

    # cv2.circle(img, center, int(math.sqrt(r_avg)), [0, 255, 255], 2)

    return img


if __name__ == '__main__':
    dataset = json.load(open(dataset_path))
    img = cv2.imread(raw_img_path)
    print(dataset['circle_radius'])
    img = stat_radius(dataset['contour'], dataset['circle_center'], dataset['circle_radius'],img)
    cv2.imshow('result', img)
    cv2.waitKey(0)
