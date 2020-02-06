import numpy as np
import cv2
import glob

catur_lebar = 20 # 26, 20.5
calib_resize = 1
# termination criteria, criteria = (func, lebar kotak dalam mm,)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, catur_lebar, 0.001)

# persiapan object point seperti (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
catur_size = (9,6)
objp = np.zeros((catur_size[0]*catur_size[1],3), np.float32)
objp[:,:2] = np.mgrid[0:catur_size[0],0:catur_size[1]].T.reshape(-1,2)

# Array untuk menyimpan object points dan image points dari semua gambar
objpoints = [] # 3d point di dunia nyata / spasial
imgpoints = [] # 2d point di bidang gambar

images = glob.glob(URL+'kalibrasi/A6000/50mm-2/edgePreserving/*.jpg')

i=0
for filename in images:
    i+=1
    img = cv.imread(filename)
    # img = cv.resize(img,(int(img.shape[1]/calib_resize),int(img.shape[0]/calib_resize))) # to resize 
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    # mencari sudut papan catur
    ret, corners = cv.findChessboardCorners(gray, catur_size,flags=cv.CALIB_CB_ADAPTIVE_THRESH) #  | cv.CALIB_CB_NORMALIZE_IMAGE | cv.CALIB_CB_FILTER_QUADS | cv.CALIB_CB_FAST_CHECK

    # nambahin object points, image points
    if ret == True:
        objpoints.append(objp)
        cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners)

        # nunjukin cornernya
        # cv.drawChessboardCorners(img,catur_size, corners,ret)
        # cv.imshow('img',img)
        # cv.waitKey()
# cv.destroyAllWindows()
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

mean_error = 0
for i in xrange(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )

# optimal new camera matrix
scaling_param = 1
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx,dist,(w,h),scaling_param,0,(w,h))

# # UNDISTORT TEST
# dst = cv.undistort(img, mtx, dist, None, newcameramtx)
# dst = cv.resize(dst,(int(img.shape[1]/calib_resize),int(img.shape[0]/calib_resize)))
# crop the image
# x, y, w, h = roi
# dst = dst[y:y+h, x:x+w]
# cv.imwrite('calibresult.png', dst)

print("mtx = "+str(mtx))
print("dist = "+str(dist))
print("dist = "+str(dist))