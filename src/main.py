import cv2
from process import *
from math_utils import *
import droplet
import folder_handler

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
    circle = circle_regression(contours[0])
    cv2.circle(contour_raw_img, circle.center, circle.radius, (0, 255, 255), 2)
    cv2.circle(contour_raw_img, circle.center, 1, (0, 255, 255), 3)

    if debug:
        # display result
        cv2.imshow('origin', img)
        cv2.imshow('gray', gray_img)
        cv2.imshow('binary', binary_img)
        cv2.imshow('denoised', denoised_img)
        cv2.imshow('contours', contour_img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # cv2.imwrite(f'../assets/output_dir/contours{i+1}.jpg', contour_img)
    cv2.imwrite(f'../assets/output_dir/contour_{i+1}.jpg', contour_raw_img)
    # save_contour(contours[0], f'../assets/output_dir/contours{i+1}_pt.csv')
    droplet.save_droplet_data(contours[0], circle, i)

if __name__ == '__main__':
    folder_path = r'/media/zhewen/d1/records/Drop Impact on Rough Surfaces/S1-W-18G-40cm-OG1-1_C001H001S0001'
    result_path = r'/media/zhewen/d1/records/Drop Impact on Rough Surfaces/results/S1-W-18G-40cm-OG1-1_C001H001S0001'
    fd = folder_handler.FolderHandler(folder_path, result_path)
    fd.exec()
    # for i in range(size):
    #     img = load_img(f'../assets/input_dir/{filename_prefix}{i+1}.jpg')
    #     background_img = load_img('../assets/input_dir/background.jpg')
    #     detect_once(img, background_img, i)
