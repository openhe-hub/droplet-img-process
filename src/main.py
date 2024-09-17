import cv2
from process import *


if __name__ == '__main__':
    id: int = 4
    img = load_img(f'../assets/input/demo{id}.jpg')
    background_img = load_img('../assets/input/background.jpg')
    diff_img = diff_img(img, background_img)
    gray_img = gbr_to_gray(diff_img)
    binary_img = gbr_to_binary(gray_img)
    denoised_img = denoise_img(binary_img)
    contours = find_contour_by_canny(denoised_img)
    contours = filter_contour(contours)
    contour_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
    contour_raw_img = img.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    cv2.drawContours(contour_raw_img, contours, -1, (0, 255, 0), 2)

    # display result
    cv2.imshow('origin', img)
    cv2.imshow('gray', gray_img)
    cv2.imshow('binary', binary_img)
    cv2.imshow('denoised', denoised_img)
    cv2.imshow('contours', contour_img)

    cv2.imwrite(f'../assets/output/contours{id}.jpg', contour_img)
    cv2.imwrite(f'../assets/output/contours{id}_on_raw.jpg', contour_raw_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()