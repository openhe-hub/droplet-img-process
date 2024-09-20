import numpy as np
from attr import dataclass
from scipy.optimize import least_squares
import cv2

@dataclass
class RegressionCircle:
    center: (int, int)
    radius: int

def circle_regression(contour: cv2.Mat) -> RegressionCircle:
    def residuals(params, x, y):
        a, b, r = params
        return np.sqrt((x - a)**2 + (y - b)**2) - r

    xs, ys = [], []
    for point in contour:
        x, y = point[0]
        xs.append(x)
        ys.append(y)

    xs, ys = np.array(xs),np.array(ys)

    initial_guess = [0,0,0]

    result = least_squares(residuals, initial_guess, args=(xs, ys))

    a, b, r = result.x
    return RegressionCircle(center=(round(a), round(b)), radius=round(r))