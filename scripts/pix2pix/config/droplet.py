## droplet
surface_type_keys = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']
surface_type_vals = [(i+1) / (len(surface_type_keys)+1) for i in range(len(surface_type_keys))]
surface_type_dict = dict(zip(surface_type_keys, surface_type_vals))

fall_point_type_keys = ['OG1', 'OG', 'OG2', 'OR1', 'OR', 'OR2', 'C', 'R', 'G']

fall_point_type_vals = [(i+1) / (len(fall_point_type_keys)+1) for i in range(len(fall_point_type_keys))]
fall_point_type_dict = dict(zip(fall_point_type_keys, fall_point_type_vals))

liquid_type_keys = ['W']
liquid_type_vals = [(i+1) / (len(fall_point_type_keys)+1)
                    for i in range(len(fall_point_type_keys))]
liquid_type_dict = dict(zip(liquid_type_keys, liquid_type_vals))

frame_id_range = [0, 40]
height_range = [10, 40]
diameter_range = [18, 25]
sm1_range = [3, 7]

def get_smooth1_data(surface_type):
    sm1 =  6.2

    if surface_type == 'S1': sm1 = 3.914
    elif surface_type == 'S2': sm1 = 7.5
    
    return (sm1 - sm1_range[0]) / (sm1_range[1] - sm1_range[0])

def get_delta_elevation(surface_type):
    delta_elevation = 0

    if surface_type == 'S4': delta_elevation = 0.33
    elif surface_type == 'S5': delta_elevation = 0.656
    elif surface_type == 'S6': delta_elevation = 1

    return delta_elevation

def analyze_conditions(path, frame_id):
    label = path.split('\\')[-1]
    seg = label.split('-')
    data = {}
    if len(seg) == 6:
        data['surface_type'] = surface_type_dict[seg[0]]
        data['liquid_type'] = liquid_type_dict[seg[1]]
        data['diameter'] = (int(seg[2][:2]) - diameter_range[0]) / (diameter_range[1] - diameter_range[0])
        data['height']  = (int(seg[3][:2]) - height_range[0]) / (height_range[1] - height_range[0])
        data['fall_point_type'] = fall_point_type_dict[seg[4]]
        data['frame_id'] = (frame_id - frame_id_range[0]) / (frame_id_range[1] - frame_id_range[0])
        data['sm1'] = get_smooth1_data(seg[0])
        data['sm2'] = 0.429
        data['delta_elevation'] = get_delta_elevation(seg[0])

    return data

def analyze_frame_id(path):
    return int(path[-10:-4])