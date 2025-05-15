import cv2
import numpy as np

def setMask(src: np.ndarray, mask: np.ndarray) -> np.ndarray:
    channels = cv2.split(src)
    result = []
    for i in range(len(channels)):
        result.append(cv2.bitwise_and(channels[i], mask))
    dest = cv2.merge(result)
    return dest

class ColorFilter(object):
    def __init__(self):
        self.colorRange = [
            ((0,43,46),(10,255,255)),
            ((156,43,46),(180,255,255)),
        ]

    def __call__(self, src: np.ndarray) -> np.ndarray:
        # 必要的函数注释
        finalMask = np.zeros_like(src)[:, :, 0]
        # finalMask指的是 最终合成的掩膜
        hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        # 将图像转为HSV通道
        for each in self.colorRange:
            # 逐一获取HSV范围
            lower, upper = each
            # HSV范围解构为 起始 和 结束
            mask = cv2.inRange(hsv, lower, upper)
            # 制作该HSV范围的掩膜
            finalMask = cv2.bitwise_or(finalMask, mask)
            # 掩膜合并 目标颜色 = 颜色1 + 颜色2 + ... + 颜色n
            # 注：inRange()不在HSV范围内的部分 数值为 0
            dest = setMask(src, finalMask)
            # 设置掩膜
            return dest