# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# def boxfilter(img, r):
#     (rows, cols) = img.shape
#     imDst = np.zeros_like(img)
#
#     imCum = np.cumsum(img, 0)
#     imDst[0 : r+1, :] = imCum[r : 2*r+1, :]
#     imDst[r+1 : rows-r, :] = imCum[2*r+1 : rows, :] - imCum[0 : rows-2*r-1, :]
#     imDst[rows-r: rows, :] = np.tile(imCum[rows-1, :], [r, 1]) - imCum[rows-2*r-1 : rows-r-1, :]
#
#     imCum = np.cumsum(imDst, 1)
#     imDst[:, 0 : r+1] = imCum[:, r : 2*r+1]
#     imDst[:, r+1 : cols-r] = imCum[:, 2*r+1 : cols] - imCum[:, 0 : cols-2*r-1]
#     imDst[:, cols-r: cols] = np.tile(imCum[:, cols-1], [r, 1]).T - imCum[:, cols-2*r-1 : cols-r-1]
#
#     return imDst
#
#
# def guidedfilter(I, p, r, eps):
#     (rows, cols) = I.shape
#     N = boxfilter(np.ones([rows, cols]), r)
#
#     meanI = boxfilter(I, r) / N
#     meanP = boxfilter(p, r) / N
#     meanIp = boxfilter(I * p, r) / N
#     covIp = meanIp - meanI * meanP
#
#     meanII = boxfilter(I * I, r) / N
#     varI = meanII - meanI * meanI
#
#     a = covIp / (varI + eps)
#     b = meanP - a * meanI
#
#     meanA = boxfilter(a, r) / N
#     meanB = boxfilter(b, r) / N
#
#     np.set_printoptions(threshold=np.inf)
#     print(meanA)
#     # print(meanB)
#     q = meanA * I + meanB
#     return q
# if __name__ == '__main__':
#     img = cv2.imread('data0003.jpg', -1)
#     guide_map = cv2.imread('img0003.tif+0+mean.jpg', -1)
#
#     out = guidedfilter(guide_map, img, 200,0.000000000000000000000000000001)
#     a=np.max(out)
#     b=np.min(out)
#     out = (255*(out-b)/(a-b)).astype(np.uint8)
#     cv2.imwrite('000sss1.png', out)
#     plt.subplot(2, 3, 1), plt.title('b')
#     plt.imshow(img, cmap='gray'), plt.axis('off')
#     plt.subplot(2, 3, 2), plt.title('b')
#     plt.imshow(guide_map, cmap='gray'), plt.axis('off')
#     plt.subplot(2, 3, 3), plt.title('b')
#     plt.imshow(out, cmap='gray'), plt.axis('off')
#     plt.show()
#
# 1.76：图像的非线性滤波—导向滤波器
# 注意：本例程需要 opencv-contrib-python 包的支持
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("/media/drh/DATA/calcium/Neighbor2Neighbor-main/data0003.jpg", flags=1)
imgGuide = cv2.imread("/media/drh/DATA/calcium/Neighbor2Neighbor-main/img0003.tif+0+mean.jpg", flags=1)  # 引导图片

imgBiFilter = cv2.bilateralFilter(img, d=0, sigmaColor=100, sigmaSpace=10)
imgGuidedFilter = cv2.ximgproc.guidedFilter(imgGuide, img, 10, 0.01, -1)

plt.figure(figsize=(9, 6))
plt.subplot(131), plt.axis('off'), plt.title("Original")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(132), plt.axis('off'), plt.title("cv2.bilateralFilter")
plt.imshow(cv2.cvtColor(imgBiFilter, cv2.COLOR_BGR2RGB))
plt.subplot(133), plt.axis('off'), plt.title("cv2.guidedFilter")
plt.imshow(cv2.cvtColor(imgGuidedFilter, cv2.COLOR_BGR2RGB))
out = imgGuidedFilter
# a=np.max(out)
# b=np.min(out)
# print(a,b)
# out = (255*(out-b)/(a-b)).astype(np.uint8)
cv2.imwrite('result_mean.png', out)
plt.tight_layout()
plt.show()
