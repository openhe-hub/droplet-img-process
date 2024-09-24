import os
import json

import cv2

import process
import math_utils
import droplet

from loguru import logger

class FolderHandler:
    def __init__(self, folder_path: str, output_folder: str, img_format: str = 'jpg', background_idx: int = 1, debug: bool = False):
        self.folder_path: str = folder_path
        self.output_folder: str = output_folder
        self.img_format: str = img_format
        self.files: [str] = self.list_img_files()
        self.img_nums: int = len(self.files)
        self.droplets: [droplet.Droplet] = []
        self.background_idx: int = background_idx
        self.background_img: cv2 =  self.set_background_img()
        self.debug: bool = debug

    def list_img_files(self) -> [str]:
        files = os.listdir(self.folder_path)
        jpg_files = [file for file in files if file.endswith(self.img_format)]
        return jpg_files

    def set_background_img(self) -> cv2.Mat:
        return process.load_img(f'{self.folder_path}/{self.files[self.background_idx]}')

    def exec(self):
        for idx, file in enumerate(self.files):
            if idx == 0: continue # skip the background one
            img = process.load_img(f'{self.folder_path}/{file}')
            self.exec_once(img, idx + 1, file)
            logger.info(f'processing {file} finished')

    def exec_once(self, img: cv2.Mat, idx: int, filename: str):
        _diff_img = process.diff_img(img, self.background_img)
        gray_img = process.gbr_to_gray(_diff_img)
        binary_img = process.gbr_to_binary(gray_img)
        denoised_img = process.denoise_img(binary_img)
        contours = process.find_contour_by_canny(denoised_img)
        contours = process.filter_contour(contours)
        if len(contours) == 0:
            logger.warning('no contour found')
            self.output_null(img, idx, filename)
            return
        contour_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
        contour_raw_img = img.copy()
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
        cv2.drawContours(contour_raw_img, contours, -1, (0, 255, 0), 2)
        circle = math_utils.circle_regression(contours[0])
        cv2.circle(contour_raw_img, circle.center, circle.radius, (0, 255, 255), 2)
        cv2.circle(contour_raw_img, circle.center, 1, (0, 255, 255), 3)

        if self.debug:
            # display result
            cv2.imshow('origin', img)
            cv2.imshow('gray', gray_img)
            cv2.imshow('binary', binary_img)
            cv2.imshow('denoised', denoised_img)
            cv2.imshow('contours', contour_img)

            cv2.waitKey(0)
            cv2.destroyAllWindows()

        cv2.imwrite(f'{self.output_folder}/contour_{filename}.jpg', contour_raw_img)
        _droplet = droplet.save_droplet_data(contours[0], circle, f'{self.output_folder}/params_{filename}.json', idx)

    def output_null(self, img: cv2.Mat, idx: int, filename: str):
        cv2.imwrite(f'{self.output_folder}/contour_{filename}.jpg', img)
        with open(f'{self.output_folder}/params_{filename}.json', 'w') as f:
            json.dump({}, f, indent=4)
