import cv2
import os
import sys
import folder_handler
from loguru import logger
from PIL import Image, ImageDraw
import json

def handle(src1, src2):    
    files1 = os.listdir(src1)
    files2 = os.listdir(src2)
    raw_files1 = sorted([file for file in files1 if file.startswith('raw')])
    raw_files2 = sorted([file for file in files2 if file.startswith('raw')])
    data_files1 = sorted([file for file in files1 if file.startswith('params')])
    data_files2= sorted([file for file in files2 if file.startswith('params')])

    # 确保两个列表长度相等
    assert len(raw_files1) == len(raw_files2), "两个文件夹中的图片数量必须相等"
    
    # 每次处理两个文件夹中的一张图片
    for i in range(len(raw_files1)):
        # 读取两个文件夹中的图片
        img1 = Image.open(os.path.join(src1, raw_files1[i]))
        img2 = Image.open(os.path.join(src2, raw_files2[i]))
        data1 = json.load(open(os.path.join(src1, data_files1[i])))
        data2 = json.load(open(os.path.join(src2, data_files2[i])))
        
        # 获取图片尺寸
        width, height = img1.size
        
        # 取左半边
        left_half_width = width // 2
        img1_left = img1.crop((0, 0, left_half_width, height))
        img2_left = img2.crop((0, 0, left_half_width, height))
        
        # 创建新图片，大小为左半边的宽度
        new_img = Image.new('RGB', (left_half_width, height))
        
        # 将两张图片的左半边重叠放置
        # 可以调整alpha值来控制混合程度
        # 将第一张图片偏向红色，第二张图片偏向绿色
        for x in range(left_half_width):
            for y in range(height):
                r1, g1, b1 = img1_left.getpixel((x, y))
                r2, g2, b2 = img2_left.getpixel((x, y))
                
                # 更严格的背景判断
                threshold_r = 200  # 红色通道阈值
                threshold_g = 150  # 绿色通道阈值
                threshold_b = 150  # 蓝色通道阈值
                

                r = int(r1 * 1.5)
                g = int(g2 * 1.5)
                b = int((b1 + b2) / 2)
                    
                r = min(255, r)
                g = min(255, g)
                b = min(255, b)
                
                new_img.putpixel((x, y), (r, g, b))

                # if (r > threshold_r and g > threshold_g and b > threshold_b):
                #     r, g, b = 255, 255, 255 
                #     new_img.putpixel((x, y), (r, g, b))
        
        # 创建绘图对象
        draw = ImageDraw.Draw(new_img)
        
        # 绘制第一个轮廓 - 使用红色
        contour1 = data1['contour']
        # 确保只使用左半边的轮廓点
        contour1 = [(x, y) for x, y in contour1 if x < left_half_width]
        if len(contour1) > 1:
            draw.line(contour1, fill=(0, 255, 0), width=2)
        
        # 绘制第二个轮廓 - 使用绿色
        contour2 = data2['contour']
        # 确保只使用左半边的轮廓点
        contour2 = [(x, y) for x, y in contour2 if x < left_half_width]
        if len(contour2) > 1:
            draw.line(contour2, fill=(255, 0, 0), width=2)
        
        # 保存结果
        output_dir = "../assets/half_analysis/output"
        os.makedirs(output_dir, exist_ok=True)
        new_img.save(os.path.join(output_dir, f'combined_{i}.jpg'))


if __name__ == '__main__':
    src1 = '../assets/half_analysis/S3-W-18G-30cm-C-1_C001H001S0001'
    src2 = '../assets/half_analysis/S3-W-18G-30cm-R-1_C001H001S0001'

    handle(src1, src2)
