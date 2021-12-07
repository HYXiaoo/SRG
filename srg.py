# -*- coding:utf-8 -*-
import cv2
import numpy as np
import RegionGrowing as RG

seed = RG.RGPoint(389,256)

def on_mouse(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('Start Mouse Position: ' + str(x) + ', ' + str(y))
        seed.x = y
        seed.y = x


#import Image
im = cv2.imread('215.jpg',cv2.IMREAD_GRAYSCALE)
im_shape = im.shape
height = im_shape[0]
width = im_shape[1]
print('the shape of image :', im_shape)

cv2.namedWindow('input')
cv2.setMouseCallback('input', on_mouse, 0, )
cv2.imshow('input',im)
cv2.waitKey()

"""
imt = im[seed.x-50: seed.x+50, seed.y-50: seed.y+50]
print(imt)
cv2.imshow('OUTIMAGE', imt)
"""

img_re = RG.regionGrowing(im, 17, seed)

# 输出图像
cv2.imshow('OUTIMAGE' , img_re)
print(img_re)
cv2.waitKey(0)
