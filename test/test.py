import os
import cv2

from src.color_filter import ColorFilter
import src.process as process

path = r'../assets/input_dir/S1-W-18G-30cm-R-1_C001H001S0001000111.jpg'
background_path = r'../assets/input_dir/background.jpg'

if __name__ == '__main__':
    src = cv2.imread(path)
    background = cv2.imread(background_path)
    diff_img = cv2.absdiff(src, background)
    # color_filter = ColorFilter()
    # dest = color_filter(diff_img)
    contours = process.find_contour_by_canny(diff_img)
    contours = process.filter_contour(contours)
    contour_raw_img = src.copy()
    cv2.drawContours(contour_raw_img, contours, -1, (0, 255, 0), 2)

    # cv2.imshow('src', src)
    # cv2.imshow('background', background)
    # cv2.imshow('diff', diff_img)
    # cv2.imshow('color', dest)
    cv2.imshow('contours', contour_raw_img)
    cv2.waitKey(0)
