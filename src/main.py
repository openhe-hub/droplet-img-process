import folder_handler
import file_utils

def detect_job():
    folder_path = r'/media/zhewen/d1/records/Drop Impact on Rough Surfaces_1/S1-W-18G-30cm-R-1_C001H001S0001'
    result_path = r'/media/zhewen/d1/records/Drop Impact on Rough Surfaces_1/result/S1-W-18G-30cm-R-1_C001H001S0001'
    fd = folder_handler.FolderHandler(folder_path, result_path)
    fd.exec()

def select_job():
    result_path = r'/media/zhewen/d1/records/Drop Impact on Rough Surfaces_1/result/S1-W-18G-30cm-R-1_C001H001S0001'
    start_idx = 113
    end_idx = 136

    file_utils.filter_files(result_path, start_idx, end_idx)
    pass

if __name__ == '__main__':
    mode = 'select'
    if mode == 'detect': detect_job()
    elif mode == 'select': select_job()


