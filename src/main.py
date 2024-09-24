import folder_handler

if __name__ == '__main__':
    folder_path = r'/media/zhewen/d1/records/Drop Impact on Rough Surfaces/S1-W-18G-40cm-OG1-1_C001H001S0001'
    result_path = r'/media/zhewen/d1/records/Drop Impact on Rough Surfaces/results/S1-W-18G-40cm-OG1-1_C001H001S0001'
    fd = folder_handler.FolderHandler(folder_path, result_path)
    fd.exec()

