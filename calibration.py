import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# persiapan object point seperti (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((3*3,3), np.float32)
objp[:,:2] = np.mgrid[0:3,0:3].T.reshape(-1,2)
catur_size = (6,9)

# Array untuk menyimpan object points dan image points dari semua gambar
objpoints = [] # 3d point di dunia nyata / spasial
imgpoints = [] # 2d point di bidang gambar

URL = '/media/ranger/01D454F10F8A7250/OneDrive - Institut Teknologi Bandung/CAMPUSS_S1/TAnjink/kalibrasi/S7/'
# images = glob.glob(URL+'*.jpg')
# images = glob.glob(URL+'cropped2.png')
# images = glob.glob(URL+'distort2.jpg')
images = glob.glob(URL+'S7kalib1.jpg')
# images = glob.glob(URL+'20190124_111024.jpg')
# images = glob.glob(URL+'worQA.jpg')
print("berhasil input 1")

i=0
for filename in images:
    i+=1
    # threshx = cv.resize(thresh, (int(imrdx.shape[1] / 3), int(imrdx.shape[0] / 3)))
    img = cv2.imread(filename)
    img = cv2.resize(img,(int(img.shape[1]/3),int(img.shape[0]/3)))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    print("berhasil input 1,5 nih " + str(i))
    # mencari sudut papan catur
    ret, corners = cv2.findChessboardCorners(gray, catur_size,flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FILTER_QUADS | cv2.CALIB_CB_FAST_CHECK)
    print("berhasil input 2 "+str(i))

    # nambahin object points, image points
    if ret == True:
        objpoints.append(objp)

        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners)
        print("berhasil input 3 " + str(i))

        # nunjukin cornernya
        cv2.drawChessboardCorners(img,catur_size, corners,ret)
        cv2.imshow('img',img)
        # cv2.waitKey(500)
        cv2.waitKey()
cv2.destroyAllWindows()
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

img = cv2.imread(URL+'S7kalib1.jpg')
h,  w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# crop gambar
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult.png',dst)

# perhitungan error kalibrasi
mean_error = 0
for i in xrange(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    tot_error += error

print "total error: ", mean_error/len(objpoints)