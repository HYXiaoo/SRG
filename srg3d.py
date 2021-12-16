# -*- coding:utf-8 -*-
import os
import cv2
import numpy as np
import RegionGrowing3d as RG3
import RGTools3d as tools3

THRES_THRESHOLD = 140  # 用于阈值化图像的阈值
THRES_MAXVAL = 255  # 用于阈值化图像的最大值
RG_THRES = 1  # 区域生长用于控制生长领域是否被划入的阈值
OUT_RADIUS = 4  # 用于闭的孔洞结构元额的外半径
IN_RADIUS = 3  # 用于闭的孔洞结构元额的内半径
PIC_FOLDER_NAME = "case_1_seg_update"  # 保存图片组的文件名
seed = RG3.RGPoint3d(275, 199, 0)  # 种子点，在UI中会被初始化


# 查看肺部图片组
def show_3dImg(window_name: str, img: np.ndarray):
    while True:
        cv2.imshow(window_name, img[:, :, seed.z])
        key = cv2.waitKey()
        if key == 13:
            break
        elif key == 93:
            if seed.z < layer - 1:
                seed.z = seed.z + 1
        elif key == 91:
            if seed.z > 0:
                seed.z = seed.z - 1


# 鼠标点击时获取图片被点击点的坐标
def on_mouse(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f'[srg3d.on_mouse]Start Mouse Position = ({y},{x})')
        seed.x = y
        seed.y = x


# 获取结构元用以进行闭操作
structuring_element = tools3.get_hole_structuring_element(OUT_RADIUS, IN_RADIUS)

# 读入图像
prefix_list = []
dir_path = ''
for root, dirs, files in os.walk(PIC_FOLDER_NAME):
    dir_path = root
    for file in files:
        t = file.split('.')
        prefix_list.append(int(t[0]))
prefix_list.sort()
img_src_list = []
img_morphed_list = []
img_thres_list = []
for index in prefix_list:
    img_src = cv2.imread(os.path.join(dir_path, f"{index}.jpg"), cv2.IMREAD_GRAYSCALE)
    img_morphed = tools3.morphology_close(img_src, structuring_element)  # 对图像进行形态学处理
    _, img_thres = cv2.threshold(img_morphed, THRES_THRESHOLD, THRES_MAXVAL, cv2.THRESH_BINARY)  # 阈值化图像
    img_src = np.expand_dims(img_src, 2)
    img_morphed = np.expand_dims(img_morphed, 2)
    img_thres = np.expand_dims(img_thres, 2)
    img_src_list.append(img_src)
    img_morphed_list.append(img_morphed)
    img_thres_list.append(img_thres)
img_src = np.concatenate(img_src_list, 2)
img_morphed = np.concatenate(img_morphed_list, 2)
img_thres = np.concatenate(img_thres_list, 2)

im_shape = img_src.shape
height = im_shape[0]
width = im_shape[1]
layer = im_shape[2]
print(f'[srg3d]the shape of image = {im_shape}')

# 显示形态学处理后的图像
cv2.namedWindow('MORPH')
print("[srg3d]morphology processed image, press enter to continue...")
show_3dImg('MORPH', img_morphed)
cv2.destroyWindow('MORPH')

# 选择生长种子点
cv2.namedWindow('INPUT')
cv2.setMouseCallback('INPUT', on_mouse, 0)
print("[srg3d]select start point and press enter to continue...")
show_3dImg('INPUT', img_thres)
print(f"[srg3d]DEBUG: scale of seed is {img_thres[seed.x, seed.y, seed.z]}")
cv2.destroyWindow('INPUT')

# -------------------debug--------------------
# print(img_thres.shape)
# cv2.imshow('area', img_thres[seed.x-50:seed.x+50, seed.y-50:seed.y+50, seed.z])
# cv2.waitKey(0)
# -------------------debug--------------------

img_re, region_point_list = RG3.regionGrowing3d(img_thres, RG_THRES, seed)

# 输出图像
print("[srg3d]press enter to show region-removed image...")
cv2.namedWindow('OUTIMAGE')
show_3dImg('OUTIMAGE', img_re)
cv2.destroyWindow('OUTIMAGE')

img_re = tools3.remove_region_by_points(img_src, region_point_list)

# 输出消除区域后的图像
print("[srg3d]press enter to exit...")
cv2.namedWindow('REMOVEDIMAGE')
show_3dImg('REMOVEDIMAGE', img_re)
print(f"[srg3d]DEBUG: scale of seed is {img_src[seed.x, seed.y, seed.z]}")
cv2.destroyWindow('REMOVEDIMAGE')

# print(img_re)
# cv2.waitKey(0)
