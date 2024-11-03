import numpy as np
import cv2
import json
from dataclasses import dataclass, asdict, field
from math_utils import RegressionCircle
from typing import List, Tuple


@dataclass
class Droplet:
    id: int = 0
    contour: List[List[int]] = field(default_factory=list)
    area: float = 0.0
    circumstance: float = 0.0
    circularity: float = 0.0
    is_fell: bool = False
    velocity: float = 0.0
    finger_num: int = 0
    finger_centers: List[List[int]] = field(default_factory=list)
    finger_lengths: List[float] = field(default_factory=list)
    circle_center: List[int] = field(default_factory=list)
    circle_radius: float = 0.0

def gen_droplet_data(contour: cv2.Mat, circle: RegressionCircle, i: int, prev_droplets: List[Droplet]
                      ,is_fell: bool) -> Droplet:
    droplet = Droplet(id=i)
    # save contour
    contour_pts = []
    for point in contour:
        x, y = point[0]
        contour_pts.append([x.item(), y.item()])
    droplet.contour = contour_pts
    # save area, circumstance, circularity
    droplet.area = calc_area(contour)
    droplet.circumstance = calc_circumstance(contour)
    droplet.circularity = 4 * np.pi * (droplet.area / np.power(droplet.circumstance, 2))
    # save regression circle
    droplet.circle_center = circle.center
    droplet.circle_radius = circle.radius
    # calc the expand velocity
    if prev_droplets == [] or prev_droplets[-1] is None or i < 30:
        droplet.velocity = 0.0
    else:
        droplet.velocity = calc_velocity(droplet.circle_radius, prev_droplets[-1].circle_radius, 1.0)
    # judge if fall
    if prev_droplets == [] or prev_droplets[-1] is None or i < 30:
        droplet.is_fell = is_fell
    else:
        droplet.is_fell = judge_if_fall(droplet.velocity, prev_droplets[-1].is_fell)

    return droplet

def save_droplet_data(droplet: Droplet, file_path):
    # dump to json
    droplet_dict = asdict(droplet)
    with open(f'{file_path}', 'w') as f:
        json.dump(droplet_dict, f, indent=4)
    return droplet

def calc_area(contour: cv2.Mat) -> float:
    return cv2.contourArea(contour)

def calc_circumstance(contour: cv2.Mat) -> float:
    return cv2.arcLength(contour, closed=True)

def calc_velocity(curr_radius: float, prev_radius: float, delta_t: float) -> float:
    return (curr_radius - prev_radius) / delta_t

def judge_if_fall(velocity: float, prev_if_fall) -> bool:
    if not prev_if_fall and velocity > 5.0:
        print(velocity)
        return True
    elif prev_if_fall:
        return True
    else:
        return False
