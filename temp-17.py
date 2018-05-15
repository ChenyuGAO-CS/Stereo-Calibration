import cv2
import numpy as np
import glob

'''left camera calibration'''

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

objp_s =[]
imgpl_s=[]

images = glob.glob('.//left//*.jpg')
#print(images)
flag = 0
cnt = 0
for fname in images:
    img = cv2.imread(fname)
    #print(img)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #print(gray)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,6),None)
    #print(corners)

    # If found, add object points, image points (after refining them)
    if ret == True:
        cnt = cnt+1
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        if flag < 7:
            objp_s.append(objp)
            flag = flag+1

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
        # cv2.imshow('img',img)
        # cv2.waitKey()

#cv2.destroyAllWindows()
#prepare image points of left folder for Stereo Calibration
imgpl_s.append(imgpoints[0]) #left01
imgpl_s.append(imgpoints[2]) #left03
imgpl_s.append(imgpoints[3]) #left04
imgpl_s.append(imgpoints[5]) #left06
imgpl_s.append(imgpoints[6]) #left07
imgpl_s.append(imgpoints[8]) #left12
imgpl_s.append(imgpoints[9]) #left13
#imgpl_s.append(imgpoints[10]) #left14


l_ret, l_mtx, l_dist, l_rvecs, l_tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
#print(gray.shape[::-1])
print("Left Camera Matrix:\n")
print(l_mtx)
print("Left Distortion Coefficients:\n")
print(l_dist)


'''right camera calibration'''
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints_r = [] # 2d points in image plane.
imgpr_s = []

images = glob.glob('.//right//*.jpg')
#print(images)
flag = 0
for fname in images:
    img = cv2.imread(fname)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (7,6),None)

    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints_r.append(corners2)

        img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
        # cv2.imshow('img',img)
        # cv2.waitKey()

#cv2.destroyAllWindows()

#prepare image points of right folder for Stereo Calibration
imgpr_s.append(imgpoints_r[0]) #right01
imgpr_s.append(imgpoints_r[1]) #right03
imgpr_s.append(imgpoints_r[2]) #right04
imgpr_s.append(imgpoints_r[3]) #right06
imgpr_s.append(imgpoints_r[4]) #right07
imgpr_s.append(imgpoints_r[7]) #right12
imgpr_s.append(imgpoints_r[8]) #right13
#imgpr_s.append(imgpoints_r[9]) #right14

r_ret, r_mtx, r_dist, r_rvecs, r_tvecs = cv2.calibrateCamera(objpoints, imgpoints_r, gray.shape[::-1],None,None)

print("Right Camera Matrix:\n")
print(r_mtx)
print("Right Distortion Coefficients:\n")
print(r_dist)

#Stereo Calibrate
ANS=cv2.stereoCalibrate(objp_s,imgpl_s,imgpr_s,l_mtx,l_dist,r_mtx,r_dist,gray.shape[::-1])
R = ANS[5]
T = ANS[6]
E = ANS[7]
F = ANS[8]
#print(ANS)
print("R:\n")
print(R)
print("T:\n")
print(T)

''' The codes before line 130 are same as the code in temp2_4.py'''

#Stereo Rectify
size = (640,480)
R1, R2, P1, P2, Q, validPixROI1, validPixROI2=cv2.stereoRectify(l_mtx,l_dist,r_mtx,r_dist,size,R,T)
# print(R1)
# print(R2)
# print(P1)
# print(P2)
left_map1, left_map2=cv2.initUndistortRectifyMap(l_mtx,l_dist,R1,P1,size,cv2.CV_16SC2)
right_map1, right_map2=cv2.initUndistortRectifyMap(r_mtx,r_dist,R2,P2,size,cv2.CV_16SC2)
#Read the image
img_l = cv2.imread('.//left//left14.jpg')
img_r = cv2.imread('.//right//right14.jpg')

imgl_rectified=cv2.remap(img_l, left_map1, left_map2, cv2.INTER_LINEAR)
imgr_rectified=cv2.remap(img_r, right_map1, right_map2, cv2.INTER_LINEAR)

#Draw two images in the same window ans draw lines
canvas = np.zeros((480,640*2,3),dtype="uint8")
green = (0, 255, 0)
red = (0, 0, 255)
print((imgl_rectified[48,0,0]))
for i in range(0,480):
    for j in range(0,640):
        #print((imgl_rectified))
        canvas[i, j, 0] = imgl_rectified[i, j, 0]
        canvas[i, j, 1] = imgl_rectified[i, j, 1]
        canvas[i, j, 2] = imgl_rectified[i, j, 2]

        canvas[i, j + 640, 0] = imgr_rectified[i, j, 0]
        canvas[i, j + 640, 1] = imgr_rectified[i, j, 1]
        canvas[i, j + 640, 2] = imgr_rectified[i, j, 2]

for i in range(2,20):
    cv2.line(canvas,(40,i*24),(640*2-20,i*24),green)
cv2.rectangle(canvas,(40,40),(620,470),red,3)
cv2.rectangle(canvas,(40+640,40),(620+640,470),red,3)

# cv2.imshow("Canvas", canvas)
# cv2.waitKey()
# cv2.destroyAllWindows()

''' The codes before line 173 are same as the code in temp-15.py'''

'''BM'''
imgL = cv2.cvtColor(imgl_rectified,cv2.COLOR_BGR2GRAY)
imgR = cv2.cvtColor(imgr_rectified,cv2.COLOR_BGR2GRAY)

num = cv2.getTrackbarPos("num", "depth")
blockSize = cv2.getTrackbarPos("blockSize", "depth")
if blockSize % 2 == 0:
    blockSize += 1
if blockSize < 5:
     blockSize = 5

stereo = cv2.StereoBM_create(numDisparities=16*num, blockSize=blockSize)
print(16*num)
print(blockSize)
#stereo = cv2.StereoBM_create(32, 11)

disparity = stereo.compute(imgL, imgR)

disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

threeD = cv2.reprojectImageTo3D(disparity.astype(np.float32) / 16., Q)

# cv2.imshow("left", imgl_rectified)
# cv2.imshow("right", imgr_rectified)
cv2.imshow("depth", disp)
cv2.waitKey()
cv2.destroyAllWindows()

'''SGBM'''
sbm = cv2.StereoSGBM_create(0,16*3, 5)
disparity_sbm = sbm.compute(imgL, imgR)
disp_sbm =cv2.normalize(disparity_sbm, disparity_sbm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# cv2.imshow("left", imgl_rectified)
# cv2.imshow("right", imgr_rectified)
cv2.imshow("depth", disp_sbm)
cv2.waitKey()
cv2.destroyAllWindows()

