### carbon mixed
def analyze_conditions(path, frame_id):
    splits = path.split('\\')[-1].split('_')[1:6]
    data = {}
    data['p_ratio'] = float(splits[0]) * 100
    data['o_ratio'] = float(splits[1]) * 100
    data['d_ratio'] = float(splits[2]) * 100
    data['temperature'] = float(splits[3].replace('C', ''))
    data['pressure'] = float(splits[4].replace('kPa', ''))
    data['frame_id'] = frame_id
    return data

def analyze_frame_id(path):
    return int(path[-6:-4])