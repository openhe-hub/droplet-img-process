import os
import json

import cv2

import process
import math_utils
import droplet

from loguru import logger

from droplet import Droplet
from finger_handler import handle_finger


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
        self.is_fell: bool = False

    def list_img_files(self) -> [str]:
        files = os.listdir(self.folder_path)
        jpg_files = sorted([file for file in files if file.endswith(self.img_format)])
        return jpg_files

    def set_background_img(self) -> cv2.Mat:
        return process.load_img(f'{self.folder_path}/{self.files[self.background_idx]}')

    def exec(self, start_idx, end_idx):
        for idx, file in enumerate(self.files):
            if idx < start_idx or idx > end_idx: continue # skip the background one
            img = process.load_img(f'{self.folder_path}/{file}')
            _droplet = self.exec_once(img, idx + 1, file.split('.')[0])
            self.droplets.append(_droplet)
            logger.info(f'processing {file} finished')

    def exec_once(self, img: cv2.Mat, idx: int, filename: str) -> Droplet:
        _diff_img = process.diff_img(img, self.background_img)
        # color_filter = ColorFilter()
        # filtered_img = color_filter(_diff_img)
        gray_img = process.gbr_to_gray(_diff_img)
        binary_img = process.gbr_to_binary(gray_img)
        denoised_img = process.denoise_img(binary_img)
        contours = process.find_contour_by_canny(denoised_img)
        contours = process.filter_contour(contours)
        if len(contours) == 0:
            logger.warning('no contour found')
            self.output_null(img, idx, filename)
            return
        # contour_img = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2BGR)
        contour_raw_img = img.copy()
        # cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
        cv2.drawContours(contour_raw_img, contours, -1, (0, 255, 0), 2)
        circle = math_utils.circle_regression(contours[0])
        cv2.circle(contour_raw_img, circle.center, circle.radius, (0, 255, 255), 2)
        cv2.circle(contour_raw_img, circle.center, 1, (0, 255, 255), 3)

        if self.debug:
            # display result
            cv2.imshow('origin', img)
            # cv2.imshow('color_filtered', filtered_img)
            cv2.imshow('contours', contour_raw_img)

            cv2.waitKey(0)
            cv2.destroyAllWindows()

        _droplet = droplet.gen_droplet_data(contours[0], circle, idx, self.droplets, self.is_fell)
        _droplet, result_img = handle_finger(_droplet, img, contour_raw_img, self.background_img)
        droplet.save_droplet_data(_droplet, f'{self.output_folder}/params_{filename}.json')
        # add text
        if _droplet is not None:
            cv2.putText(result_img, f'v={_droplet.velocity}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(result_img, f'fall={_droplet.is_fell}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        #
        self.is_fell = _droplet.is_fell
        cv2.imwrite(f'{self.output_folder}/raw_{filename}.jpg', img)
        cv2.imwrite(f'{self.output_folder}/result_{filename}.jpg', result_img)
        return _droplet

    def output_null(self, img: cv2.Mat, idx: int, filename: str):
        cv2.imwrite(f'{self.output_folder}/contour_{filename}.jpg', img)
        with open(f'{self.output_folder}/params_{filename}.json', 'w') as f:
            json.dump({}, f, indent=4)
