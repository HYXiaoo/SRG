import numpy as np
import os
import cv2

radius = 2
iner_radius = 1
kernelo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
kernerl = np.ones((int(radius * 2), int(radius * 2))).astype(np.uint8)
print(type(kernerl[0, 0]))

numr, numc = kernerl.shape

for i in range(numr):
    for j in range(numc):
        if iner_radius ** 2 <= (i - radius + 0.5) ** 2 + (j - radius + 0.5) ** 2 <= radius ** 2:
            kernerl[i, j] = 0

print(kernerl)

im = cv2.imread("./case_1_seg_update/108.jpg", cv2.IMREAD_GRAYSCALE)
# im_re = cv2.morphologyEx(im,cv2.MORPH_OPEN,kernelo)
im_re = cv2.medianBlur(im, 3)
im_re2 = cv2.morphologyEx(im_re, cv2.MORPH_OPEN, kernerl)

cv2.imshow('show', im)
cv2.waitKey(0)
cv2.imshow('res', im_re)
cv2.waitKey(0)
cv2.imshow('res', im_re2)
cv2.waitKey(0)
