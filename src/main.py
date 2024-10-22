import folder_handler

if __name__ == '__main__':
    folder_path = r'/media/zhewen/d1/records/Drop Impact on Rough Surfaces_1/S1-W-18G-30cm-R-1_C001H001S0001'
    result_path = r'/media/zhewen/d1/records/Drop Impact on Rough Surfaces_1/result/S1-W-18G-30cm-R-1_C001H001S0001'
    fd = folder_handler.FolderHandler(folder_path, result_path)
    fd.exec()

