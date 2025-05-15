import folder_handler
from dataset_handler import DatasetHandler
import os
import json
from loguru import logger
import pandas as pd

def detect_job(path, folder_name, start_idx, end_idx):
    logger.success(f'=== Detect: {folder_name} ===')
    folder_path = f'{path}\\{folder_name}'
    result_path = f'{path}\\result\\{folder_name}'
    # os.mkdir(result_path)
    fd = folder_handler.FolderHandler(folder_path, result_path, img_format='png', background_idx=0)
    fd.exec(start_idx, end_idx)


def dataset_gen_job(path, folder_name):
    logger.success(f'=== Gen Dataset: {folder_name} ===')
    result_path = f'{path}\\result\\{folder_name}'
    dataset_handler = DatasetHandler(result_path)
    dataset_handler.gen_dataset()

def combine(path, files):
    dfs = []
    file_paths = [f'{path}\\result\\{file}\\result.csv' for file in files]

    for file_path in file_paths:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.to_csv(f'{path}\\result\\result.csv', index=False)
    combined_df.to_excel(f'{path}\\result\\result.xlsx', index=False)

def combine2(folders):
    dfs = []
    file_paths = [f'D:\\records\\{folder}\\result\\result.csv' for folder in folders]

    for file_path in file_paths:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            dfs.append(df)
        else:
            logger.warning(f'{file_path} not found.')

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.to_csv(f'../assets/dataset.csv', index=False)


if __name__ == '__main__':
    detect_job(r'E:\program\python\droplet-img-process\assets','images_pred', 1, 100)
    dataset_gen_job(r'E:\program\python\droplet-img-process\assets','images_pred')


    # config_path = '../config/seg_config.json'
    # config = json.load(open(config_path))
    # folders = list(config.keys())
    # combine2(folders)
    # segs = config[folder_path.split('\\')[-1]]['seg']
    # files = os.listdir(folder_path)
    # files = sorted([file for file in files if file.startswith('S')])
    # for file in files:
    #     # if not seg: continue
    #     print(file)
    #     detect_job(folder_path, file, 90, 140)
    #     dataset_gen_job(folder_path, file)
    # combine(folder_path, files)
