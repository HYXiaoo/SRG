import numpy as np


# 点类，方便进行处理
class RGPoint(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def getX(self):
        return self.x

    def getY(self):
        return self.y


# 8邻域
connects = [RGPoint(-1, -1), RGPoint(0, -1), RGPoint(1, -1),
            RGPoint(1, 0), RGPoint(1, 1), RGPoint(0, 1),
            RGPoint(-1, 1), RGPoint(-1, 0)]


# 计算图像两个点间的欧式距离
def get_dist(img: np.ndarray, seed_location1, seed_location2, is_gray_scale=True):
    l1 = img[seed_location1.x, seed_location1.y]
    l2 = img[seed_location2.x, seed_location2.y]
    count = abs(int(l1) - int(l2)) if is_gray_scale else np.sqrt(np.sum(np.square(l1 - l2)))
    # print(f"count={count}")
    return count


# 区域生长算法
def regionGrowing(img: np.ndarray,
                  threshold: float,  # 阈值
                  seed: RGPoint,  # 种子点
                  is_gray_scale=True
                  ):
    height = img.shape[0]
    width = img.shape[1]

    # 标记，判断种子是否已经生长
    img_mark = np.zeros([height, width])

    # 建立空的图像数组,作为一类
    img_re = img.copy()
    for i in range(height):
        for j in range(width):
            if is_gray_scale:
                img_re[i, j] = 0
            else:
                img_re[i, j][0] = 0
                img_re[i, j][1] = 0
                img_re[i, j][2] = 0
    # push种子点
    # print(f"seed=({seed.x},{seed.y})")
    seed_list = [seed]
    class_k = 1  # 类别
    # 生长一个类
    while len(seed_list) > 0:
        seed_tmp = seed_list[0]
        # 将已经生长的点从一个类的种子点列表中删除
        seed_list.pop(0)

        img_mark[seed_tmp.x, seed_tmp.y] = class_k

        # 遍历8邻域
        for i in range(8):
            tmpX = seed_tmp.x + connects[i].x
            tmpY = seed_tmp.y + connects[i].y

            if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= width:
                continue

            dist = get_dist(img, seed_tmp, RGPoint(tmpX, tmpY))

            # 在种子集合中满足条件的点进行生长
            if dist < threshold and img_mark[tmpX, tmpY] == 0:
                if is_gray_scale:
                    img_re[tmpX, tmpY] = img[tmpX, tmpY]
                else:
                    img_re[tmpX, tmpY][0] = img[tmpX, tmpY][0]
                    img_re[tmpX, tmpY][1] = img[tmpX, tmpY][1]
                    img_re[tmpX, tmpY][2] = img[tmpX, tmpY][2]
                img_mark[tmpX, tmpY] = class_k
                seed_list.append(RGPoint(tmpX, tmpY))

    return img_re
