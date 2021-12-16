import numpy as np


class RGPoint3d:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


# 计算图像两个点间的欧式距离
def get_dist(img: np.ndarray, seed_location1, seed_location2, is_gray_scale=True):
    l1 = img[seed_location1.x, seed_location1.y, seed_location1.z]
    l2 = img[seed_location2.x, seed_location2.y, seed_location2.z]
    count = abs(int(l1) - int(l2)) if is_gray_scale else np.sqrt(np.sum(np.square(l1 - l2)))
    # print(f"count={count}")
    return count


# 区域生长算法
# 返回生长后的区域图像数组，生长区域包含点的列表
def regionGrowing3d(img: np.ndarray,
                    threshold: float,  # 阈值
                    seed: RGPoint3d,  # 种子点
                    is_gray_scale=True
                    ):
    height = img.shape[0]
    width = img.shape[1]
    layer = img.shape[2]

    # 标记，判断种子是否已经生长
    img_mark = np.zeros([height, width, layer])

    # 建立空的图像数组,作为一类
    # -------------------旧代码-------------------
    # img_re = img.copy()
    # for i in range(height):
    #     for j in range(width):
    #         for k in range(layer):
    #             if is_gray_scale:
    #                 img_re[i, j, k] = 0
    #             else:
    #                 img_re[i, j, k][0] = 0
    #                 img_re[i, j, k][1] = 0
    #                 img_re[i, j, k][2] = 0
    # -------------------旧代码-------------------
    # -------------------新代码-------------------
    img_re = np.zeros(img.shape)
    # -------------------新代码-------------------

    # push种子点
    # print(f"seed=({seed.x},{seed.y},{seed.z})")
    seed_list = [seed]
    class_k = 1  # 类别
    list_region_points = []  # 储存生长点的列表
    # 生长一个类
    while len(seed_list) > 0:
        seed_tmp = seed_list[0]
        # 保存将被弹出的种子点到列表
        list_region_points.append(seed_tmp)
        # 将已经生长的点从一个类的种子点列表中删除
        seed_list.pop(0)

        img_mark[seed_tmp.x, seed_tmp.y, seed_tmp.z] = class_k

        # 遍历邻域
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    if i or j or k:
                        tmpX = seed_tmp.x + i
                        tmpY = seed_tmp.y + j
                        tmpZ = seed_tmp.z + k

                        if tmpX < 0 or tmpY < 0 or tmpZ < 0 or tmpX >= height or tmpY >= width or tmpZ >= layer:
                            continue

                        dist = get_dist(img, seed_tmp, RGPoint3d(tmpX, tmpY, tmpZ))
                        print(f"[regionGrowing3d.regionGrowing3d]point stack height = {len(seed_list)}")

                        # 在种子集合中满足条件的点进行生长
                        if dist < threshold and img_mark[tmpX, tmpY, tmpZ] == 0:
                            if is_gray_scale:
                                img_re[tmpX, tmpY, tmpZ] = img[tmpX, tmpY, tmpZ]
                            else:
                                img_re[tmpX, tmpY, tmpZ][0] = img[tmpX, tmpY, tmpZ][0]
                                img_re[tmpX, tmpY, tmpZ][1] = img[tmpX, tmpY, tmpZ][1]
                                img_re[tmpX, tmpY, tmpZ][2] = img[tmpX, tmpY, tmpZ][2]
                            img_mark[tmpX, tmpY, tmpZ] = class_k
                            seed_list.append(RGPoint3d(tmpX, tmpY, tmpZ))

    return img_re, list_region_points
