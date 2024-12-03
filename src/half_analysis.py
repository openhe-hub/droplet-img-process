import cv2
import os
import sys
import folder_handler
from loguru import logger

def split(path: str):
  image = cv2.imread(path)
  height, width = image.shape[:2]
  middle = width // 2
  left_half = image[:, :middle]
  right_half = image[:, middle:]
  folder_path = os.path.dirname(path)
  filename = os.path.splitext(os.path.basename(path))[0]
  cv2.imwrite(f'{folder_path}/raw/{filename}_left.jpg', left_half)
  cv2.imwrite(f'{folder_path}/raw/{filename}_right.jpg', right_half)

def detect_job(path, folder_name, start_idx, end_idx):
    logger.success(f'=== Detect: {folder_name} ===')
    folder_path = f'{path}/{folder_name}'
    result_path = f'{path}/result/{folder_name}'
    if not os.path.exists(result_path):
      os.mkdir(result_path)
    fd = folder_handler.FolderHandler(folder_path, result_path, debug=False,      background_idx=0)
    fd.exec(start_idx, end_idx)

def demo():
   raw = cv2.imread('../assets/half_analysis/input.jpg')
   left = cv2.imread('../assets/half_analysis/raw_left/input_left.jpg')
   right = cv2.imread('../assets/half_analysis/raw_right/input_right.jpg')
   left_res = cv2.imread('../assets/half_analysis/result/raw_left/result_input_left.jpg')
   right_res = cv2.imread('../assets/half_analysis/result/raw_right/result_input_right.jpg')

   cv2.imshow("raw", raw)
   cv2.imshow("split_left", left)
   cv2.imshow("split_right", right)
   cv2.imshow("result_left", left_res)
   cv2.imshow("result_right", right_res)
   cv2.waitKey(0)

if __name__ == '__main__':	
  # split('../assets/half_analysis/background.jpg')
  # detect_job('../assets/half_analysis', 'raw_left', 1, 1)
  # detect_job('../assets/half_analysis', 'raw_right', 1, 1)
  demo()

