import folder_handler
from dataset_handler import DatasetHandler
import os
from loguru import logger
import pandas as pd

def detect_job(path, folder_name, start_idx, end_idx):
    logger.success(f'=== Detect: {folder_name} ===')
    folder_path = f'{path}/{folder_name}'
    result_path = f'{path}/result/{folder_name}'
    os.mkdir(result_path)
    fd = folder_handler.FolderHandler(folder_path, result_path)
    fd.exec(start_idx, end_idx)


def dataset_gen_job(path, folder_name):
    logger.success(f'=== Gen Dataset: {folder_name} ===')
    result_path = f'{path}/result/{folder_name}'
    dataset_handler = DatasetHandler(result_path)
    dataset_handler.gen_dataset()

def combine(path, files):
    dfs = []
    file_paths = [f'{path}/result/{file}/result.csv' for file in files]

    for file_path in file_paths:
        df = pd.read_csv(file_path)
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.to_csv(f'{path}/result.csv', index=False)


if __name__ == '__main__':
    folder_path = '/media/zhewen/d1/records/Drop Impact on Rough Surfaces_1'
    files = os.listdir(folder_path)
    files = [file for file in files if file.startswith('S')]
    combine(folder_path, files)
    # for file in files:
        # detect_job(folder_path, file, 50, 500)
        # dataset_gen_job(folder_path, file)
