### carbon pure
carbon_num_dict = {
    'Methane': 1,
    'Ethane': 2,
    'Propane': 3,
    'Butane': 4,
    'Pentane': 5,
    'Hexane': 6,
    'Heptane': 7,
    'Octane': 8,
    'Nonane': 9,
    'Decane': 10,
}

def analyze_conditions(path, frame_id):
    splits = path.split('\\')[-1].split('_')[:3]
    data = {}
    data['carbon_num'] = carbon_num_dict[splits[0].capitalize()]
    data['temperature'] = float(splits[1].replace('C', ''))
    data['pressure'] = float(splits[2].replace('kPa', ''))
    data['frame_id'] = frame_id
    return data

def analyze_frame_id(path):
    return int(path[-6:-4])