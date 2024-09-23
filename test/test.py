import os

path = r'/media/zhewen/d1/records/Drop Impact on Rough Surfaces/S1-W-18G-40cm-OG1-1_C001H001S0001'

if __name__ == '__main__':
    files = os.listdir(path)
    jpg_files = [file for file in files if file.endswith('.jpg')]
    print(jpg_files)