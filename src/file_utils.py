import os

def filter_files(folder_path, start_idx, end_idx):
    for filename in os.listdir(folder_path):
        try:
            file_index = int(filename.split('.')[0][-4:])
            if not (start_idx <= file_index <= end_idx):
                file_path = os.path.join(folder_path, filename)
                os.remove(file_path)
        except ValueError:
            continue