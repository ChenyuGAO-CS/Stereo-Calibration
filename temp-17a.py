import cv2
import numpy as np
import glob

'''left camera calibration'''

#Read the image
img_l = cv2.imread('.//ALL-2views//Lampshade1//view1.png')
img_r = cv2.imread('.//ALL-2views//Lampshade1//view5.png')

'''BM'''
imgL = cv2.cvtColor(img_l,cv2.COLOR_BGR2GRAY)
imgR = cv2.cvtColor(img_r,cv2.COLOR_BGR2GRAY)

stereo = cv2.StereoBM_create(0, 5)
#stereo = cv2.StereoBM_create(32, 11)

disparity = stereo.compute(imgL, imgR)

disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

cv2.imshow("depth", disp)
cv2.waitKey()
cv2.destroyAllWindows()

'''SGBM'''
sbm = cv2.StereoSGBM_create(0,16*3, 5)
disparity_sbm = sbm.compute(imgL, imgR)
disp_sbm =cv2.normalize(disparity_sbm, disparity_sbm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

cv2.imshow("depth", disp_sbm)
cv2.waitKey()
cv2.destroyAllWindows()

