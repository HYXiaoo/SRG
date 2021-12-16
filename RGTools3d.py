import numpy as np
import cv2

FILL_GRAY_SCALE = 88  # 用于填充被去除区域的灰度值


# 去除原图像中被分割出的区域
# 返回处理后的图像数组
def remove_region(
        src_pics: np.ndarray,  # 原始3d图像
        region_pics: np.ndarray,  # 分割出区域的二值图像
        region_scale=255  # 区域在区域二值图像中的灰度标志
):
    height, width, layer = src_pics.shape
    handle_pics = src_pics.copy()

    for i in range(height):
        for j in range(width):
            for k in range(layer):
                if region_pics[i, j, k] == region_scale:
                    handle_pics[i, j, k] = FILL_GRAY_SCALE
        print(f'[RGTools3d.remove_region]processed{i}/{height}')

    # 最后进行形态学开操作去掉细小孔洞
    s_elem = get_hole_structuring_element(2, 1)
    for k in range(layer):
        handle_pics[:, :, k] = cv2.morphologyEx(handle_pics[:, :, k], cv2.MORPH_OPEN, s_elem)
        print(f'[RGTools3d.remove_region]morphology processed{k}/{layer}')

    return handle_pics


# 去除原图像中被分割出的区域, 通过区域点列表进行，处理过程被加速
# 返回处理后的图像数组
def remove_region_by_points(
        src_pics: np.ndarray,  # 原始3d图像
        region_point_list: list,  # 区域包含点的列表
):
    list_len = len(region_point_list)
    handle_pics = src_pics.copy()

    for i, point in enumerate(region_point_list):
        handle_pics[point.x, point.y, point.z] = FILL_GRAY_SCALE
        print(f'[RGTools3d.remove_region_by_points]processed{i}/{list_len}')

    # 最后进行形态学开操作去掉细小孔洞
    # s_elem = get_hole_structuring_element(2, 1)
    s_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for k in range(src_pics.shape[2]):
        handle_pics[:, :, k] = cv2.morphologyEx(handle_pics[:, :, k], cv2.MORPH_OPEN, s_elem)
        print(f'[RGTools3d.remove_region]morphology processed{k}/{src_pics.shape[2]}')

    return handle_pics


# 对2d图像数组进行形态学处理
# 返回处理后的图像数组
def morphology_close(
        image: np.ndarray,  # 2d图像数组
        s_elemet: np.ndarray,  # 2d结构元
        blur_radius=3  # 均值滤波半径
):
    image = cv2.medianBlur(image, blur_radius)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, s_elemet)


# 获取空洞结构元
# 返回结构元数组
def get_hole_structuring_element(
        outter_radius,  # 外半径
        inner_radius  # 内半径
):
    kernerl = np.ones((outter_radius * 2, outter_radius * 2)).astype(np.uint8)
    numr, numc = kernerl.shape
    for i in range(numr):
        for j in range(numc):
            if inner_radius ** 2 <= (i - outter_radius + 0.5) ** 2 + (
                    j - outter_radius + 0.5) ** 2 <= outter_radius ** 2:
                kernerl[i, j] = 0
    print(f"[RGTools3d.get_hole_structuring_element]structuring element got:\n{kernerl}")
    return kernerl
