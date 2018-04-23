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

    # If found, add object points, image points (after refining them)
    if ret == True:
        cnt = cnt+1
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        if flag < 8:
            objp_s.append(objp)
            flag = flag+1

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
        #cv2.imshow('img',img)
        #cv2.waitKey(500)

#cv2.destroyAllWindows()
#print(imgpoints[0])
imgpl_s.append(imgpoints[0]) #left01
imgpl_s.append(imgpoints[2]) #left03
imgpl_s.append(imgpoints[3]) #left04
imgpl_s.append(imgpoints[5]) #left06
imgpl_s.append(imgpoints[6]) #left07
imgpl_s.append(imgpoints[8]) #left12
imgpl_s.append(imgpoints[9]) #left13
imgpl_s.append(imgpoints[10]) #left14


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
        #cv2.imshow('img',img)
        #cv2.waitKey(500)

#cv2.destroyAllWindows()
imgpr_s.append(imgpoints_r[0]) #right01
imgpr_s.append(imgpoints_r[1]) #right03
imgpr_s.append(imgpoints_r[2]) #right04
imgpr_s.append(imgpoints_r[3]) #right06
imgpr_s.append(imgpoints_r[4]) #right07
imgpr_s.append(imgpoints_r[7]) #right12
imgpr_s.append(imgpoints_r[8]) #right13
imgpr_s.append(imgpoints_r[9]) #right14

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
print("E:\n")
print(E)
print("F:\n")
print(F)