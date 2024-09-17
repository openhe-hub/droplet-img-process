import cv2
from process import *


if __name__ == '__main__':
    img = load_img('../assets/input/demo.jpg')
    gray_img = gbr_to_gray(img)
    denoised_img = denoise_img(gray_img)
    contours = find_contour_by_canny(denoised_img)
    contours = filter_contour(contours)
    contour_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    
    # display result
    cv2.imshow('origin', img)
    cv2.imshow('gray', gray_img)
    cv2.imshow('denoised', denoised_img)
    cv2.imshow('contours', contour_img)

    cv2.imwrite('../assets/output/contours.jpg', contour_img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()