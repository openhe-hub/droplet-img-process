import cv2
from process import *

# params
debug = False
size = 4
filename_prefix = 'demo'

def detect_once(img: cv2.Mat, background_img: cv2.Mat, i: int):
    _diff_img = diff_img(img, background_img)
    gray_img = gbr_to_gray(_diff_img)
    binary_img = gbr_to_binary(gray_img)
    denoised_img = denoise_img(binary_img)
    contours = find_contour_by_canny(denoised_img)
    contours = filter_contour(contours)
    contour_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
    contour_raw_img = img.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    cv2.drawContours(contour_raw_img, contours, -1, (0, 255, 0), 2)

    if debug:
        # display result
        cv2.imshow('origin', img)
        cv2.imshow('gray', gray_img)
        cv2.imshow('binary', binary_img)
        cv2.imshow('denoised', denoised_img)
        cv2.imshow('contours', contour_img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    cv2.imwrite(f'../assets/output_dir/contours{i+1}.jpg', contour_img)
    cv2.imwrite(f'../assets/output_dir/contours{i+1}_on_raw.jpg', contour_raw_img)

if __name__ == '__main__':
    for i in range(size):
        img = load_img(f'../assets/input_dir/{filename_prefix}{i+1}.jpg')
        background_img = load_img('../assets/input_dir/background.jpg')
        detect_once(img, background_img, i)
