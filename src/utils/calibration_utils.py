import cv2
import numpy as np
import glob

# params
chessboard_size = (3, 3)
square_size = 0.005

# path
path = '/media/zhewen/d1/records/Drop Impact on Rough Surfaces_1/Calibration 1_C001H001S0001'


object_points = []
image_points = []

objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp = objp * square_size
shape = None

images = glob.glob(f'{path}/*.jpg')
print(len(images))

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    shape = gray.shape[::-1]

    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        object_points.append(objp)
        image_points.append(corners)

        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow('Chessboard Corners', img)
        cv2.waitKey(0)
    else:
        print('ERR: unable to find the chessboard corners')

cv2.destroyAllWindows()

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, shape, None, None)

fx = camera_matrix[0, 0]
pixel_size = square_size / fx

print("CAMERA MATRIX:\n", camera_matrix)
print("DIFF COEFS:\n", dist_coeffs)
print("PIXEL IN METER: {:.6f}".format(pixel_size))
