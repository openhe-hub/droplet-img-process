import pandas as pd
import numpy as np
from loguru import logger
import json
import os

# params
frame_in_sec = 0.0002
px_in_meter = 0.000012


class DatasetHandler:
    def __init__(self, path: str):
        self.path: str = path
        self.files: [str] = self.list_json_files()
        self.length = len(self.files)

    def list_json_files(self) -> [str]:
        files = os.listdir(self.path)
        files = sorted([file for file in files if file.endswith('json')])
        return files

    def analyze_conditions(self, df: pd.DataFrame) -> pd.DataFrame:
        label  = self.path.split('\\')[-1]
        seg = label.split('-')
        if len(seg) == 6:
            df['surface_type'] = seg[0]
            df['liquid_type'] = seg[1]
            df['diameter'] = seg[2][:2]
            df['height']  = seg[3][:2]
            df['fall_point_type'] = seg[4]

        # init other cols
        df['id'] = 0
        df['time'] = 0
        df['area'] = 0.0
        df['circumstance'] = 0.0
        df['circularity'] = 0.0
        df['finger_num'] = 0.0
        df['velocity'] = 0.0
        df['circle_radius'] = 0.0
        df['src'] = ""

        return df

    def fill_data(self, df: pd.DataFrame) -> pd.DataFrame:
        for idx, row in df.iterrows():
            file = self.files[idx]
            dataset = json.loads(open(f'{self.path}/{file}').read())
            if 'id' not in dataset: continue
            df.loc[idx, 'id'] = dataset['id']
            df.loc[idx, 'src'] = dataset['src']
            df.loc[idx, 'time'] = dataset['id'] * frame_in_sec
            df.loc[idx, 'area'] = dataset['area'] * (px_in_meter ** 2)
            df.loc[idx, 'circumstance'] = dataset['circumstance'] * px_in_meter
            df.loc[idx, 'circularity'] = dataset['circularity']
            df.loc[idx, 'finger_num'] = dataset['finger_num']
            df.loc[idx, 'velocity'] = dataset['velocity'] * px_in_meter / frame_in_sec
            df.loc[idx, 'circle_radius'] = dataset['circle_radius'] * px_in_meter

        return df

    def filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[df['circumstance'] > 0.01]
        df= df[df['circularity'] > 0.1]
        df= df[(df['velocity'] < 100) & (df['velocity'] > -100)]
        return df

    def gen_dataset(self):
        df = pd.DataFrame(index=range(self.length))
        df = self.analyze_conditions(df)
        df = self.fill_data(df)
        # df= self.filter_data(df)
        self.save_dataset(df)

    def save_dataset(self, df: pd.DataFrame):
        df.to_csv(f'{self.path}/result.csv', index=False)
        df.to_excel(f'{self.path}/result.xlsx', index=False)

    def stat_dataset(self):
        pass

