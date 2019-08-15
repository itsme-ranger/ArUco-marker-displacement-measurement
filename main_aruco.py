from __future__ import print_function
from __future__ import division
from numpy.linalg import pinv
import argparse
import cv2 as cv
import cv2.aruco as aruco
import shapedetector
import math
import numpy as np
from matplotlib.widgets import Slider, RadioButtons
from matplotlib import pyplot as plt
import imutils
import operator
import glob
import datetime

def cornerstoWP(corners,rvec,tvec):
    # cor = corners[0][0]
    corners = np.insert(corners, 2, 1, axis=1)
    rotmat, jacob = cv.Rodrigues(rvec)
    matr = np.append(rotmat.transpose(), tvec[0], axis=0).transpose()
    WP = np.zeros([4, 4])
    for i in range(len(corners)):
        WP[i] = np.matmul(pinv(matr), corners[i])
    return WP

def WPofCenter(WP):
    WP_center = np.zeros(3)
    ma = (WP[2][1]-WP[0][1])/(WP[2][0]-WP[0][0])
    mb = (WP[1][1]-WP[3][1])/(WP[1][0]-WP[3][0])
    WP_center[0] = (WP[3][1]-WP[0][1]+ma*WP[0][0]-mb*WP[3][0])/(ma-mb)
    WP_center[1] = ma*(WP_center[0]-WP[0][0])+WP[0][1]
    WP_center[2] = (WP[2][2]-WP[0][2])/(WP[2][0]-WP[0][0])*(WP_center[0]-WP[0][0])+WP[0][2]
    return WP_center

def distance(start, target, mat):
    sum = 0
    for x in mat:
        sum += x
    distnce = 0
    if sum != 1:
        for i in range(len(start)):
            distnce += mat[i]*(target[i]-start[i])**2
        return math.sqrt(distnce)
    else:
        for i in range(len(start)):
            distnce += mat[i]*(target[i]-start[i])
        return distnce

URL = '/media/ranger/01D454F10F8A7250/OneDrive - Institut Teknologi Bandung/CAMPUSS_S1/TAnjink/'
URLsample = 'lab/try13/edgePreserving4/'
# URLsample = 'lab/try13/'
URLstrip = ['-' if x == '/' else x for x in URLsample]
URLstrip = ''.join(URLstrip)
now = datetime.datetime.now()
# f = open("tabel translasi.txt","a+")
# f = open("sampah.txt","a+")
f = open("invoice "+URLstrip+' '+str(now.strftime("%m-%d_%H-%M"))+".txt","a+")
db = open("db "+URLstrip+' '+str(now.strftime("%m-%d_%H-%M"))+".csv","a+")
# db = open("db.csv","a+")
db.write('\n\n'+str(URLsample)+'\n'+str(datetime.datetime.now())+'\n')
f.write("\n\n----------xxxxxxx------xxxxxx-------\n\n\n")
f.write(str(datetime.datetime.now()))
f.write(str(now.strftime("%m-%d_%H-%M")))

'''/
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

# images = glob.glob(URL+'*.jpg')
# images = glob.glob(URL+'cropped2.png')
# images = glob.glob(URL+'distort2.jpg')
# images = glob.glob(URL+'kalibrasi/S7/2.jpg')
images = glob.glob(URL+'kalibrasi/A6000/50mm-2/edgePreserving/*.jpg')
# images = glob.glob(URL+'20190124_111024.jpg')
# images = glob.glob(URL+'worQA.jpg')
print("berhasil input 1")

i=0
for filename in images:
    i+=1
    # threshx = cv.resize(thresh, (int(imrdx.shape[1] / 3), int(imrdx.shape[0] / 3)))
    img = cv.imread(filename)
    # img = cv.resize(img,(int(img.shape[1]/calib_resize),int(img.shape[0]/calib_resize)))
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    print("berhasil input 1,5 nih " + str(i))
    # mencari sudut papan catur
    ret, corners = cv.findChessboardCorners(gray, catur_size,flags=cv.CALIB_CB_ADAPTIVE_THRESH) #  | cv.CALIB_CB_NORMALIZE_IMAGE | cv.CALIB_CB_FILTER_QUADS | cv.CALIB_CB_FAST_CHECK
    print("berhasil input 2 "+str(i))

    # nambahin object points, image points
    if ret == True:
        objpoints.append(objp)

        cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners)
        print("berhasil input 3 " + str(i))

        # nunjukin cornernya
        # cv.drawChessboardCorners(img,catur_size, corners,ret)
        # cv.imshow('img',img)
        # cv.waitKey()
# cv.destroyAllWindows()
# img = cv.imread(URL+'S7kalib1.jpg')
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

mean_error = 0
for i in xrange(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )
/'''
# intrinsic matrix preset
'''/ # S7
dist = [[-0.06034095,  0.30321576, -0.00254214, -0.00410065, -0.79490111]]
mtx = [[1.06049422e+03, 0.00000000e+00, 6.62591355e+02],
 [0.00000000e+00, 1.06166045e+03, 5.25901169e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
/'''
'''/ # S7
dist = [[-0.04591939, 0.08397314, -0.00252886, -0.00562079, -0.35448564]]
mtx = [[3.25067113e+03, 0.00000000e+00, 1.98146254e+03],
 [0.00000000e+00, 3.25153553e+03, 1.60561074e+03],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
newcameramtx = [[2.49414941e+03, 0.00000000e+00, 1.86592383e+03],
 [0.00000000e+00, 2.50528125e+03, 1.52372792e+03],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
/'''
'''/ # S7
dist = [[-0.0109257,  -0.00562112, -0.0041547, 0.00253377, -0.07761259]]
mtx = [[3.16828558e+03, 0.00000000e+00, 2.02424604e+03],
 [0.00000000e+00, 3.16929445e+03, 1.54754570e+03],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
newcameramtx = [[3.13358276e+03, 0.00000000e+00, 2.03524505e+03],
 [0.00000000e+00, 3.15857617e+03, 1.53856214e+03],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
/'''
# A6000 50mm -1
'''/
dist= [[-1.43041527e-02, 8.12095746e-01, -7.80905269e-03,  1.84408618e-03,  -1.64054430e+01]]
mtx = [[1.29403749e+04, 0.00000000e+00, 3.08468496e+03],
 [0.00000000e+00, 1.29339323e+04, 1.77669904e+03],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
newcameramtx = [[1.28332451e+04, 0.00000000e+00, 3.09175089e+03],
 [0.00000000e+00, 1.28451738e+04, 1.76099117e+03],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
/'''
# A6000 50mm -2 after edgepreserving1
dist = [[ 9.53306186e-03, -5.61930848e-02, -8.66880870e-03, -1.60162599e-03, -5.23222839e+00]]
mtx = [[1.30822305e+04, 0.00000000e+00, 2.95153154e+03],
 [0.00000000e+00, 1.30659778e+04, 1.68524545e+03],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
newcameramtx = [[1.30038496e+04, 0.00000000e+00, 2.94683246e+03],
 [0.00000000e+00, 1.30000098e+04, 1.67097140e+03],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
# '''/
# , dtype=np.float32)
dist = np.asarray(dist)
mtx = np.asarray(mtx)
newcameramtx = np.asarray(newcameramtx)
# mtx*=calib_resize
# dist*=calib_resize
# /'''
print(dist)
print(mtx)
# img = cv.imread(URL+'kalibrasi/S7/3/camera_1562648291872.jpg')
# imgblack = np.zeros(img.shape)
# img = cv.imread(filename)
# h,  w = img.shape[:2]
# h,  w = [4000, 6000]
# newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
print (newcameramtx)
# undistort
# dst = cv.undistort(img, mtx, dist, None, newcameramtx)
# crop the image
# x, y, w, h = roi
# dst = dst[y:y+h, x:x+w]
# dst = cv.resize(dst,(int(dst.shape[1]/3),int(dst.shape[0]/3)))
# cv.imshow("dst",dst)
# cv.waitKey()

# img = aruco.drawAxis(img,mtx,dist,rvecs[0],tvecs[0],2.5)
# cv.imshow('img',img)
# cv.waitKey()

# h,  w = img.shape[:2]
# newcameramtx = mtx
# newcameramtx, roi=cv.getOptimalNewCameraMatrix(mtx,dist,(w,h),0,(w,h))

# # undistort
# dst = cv.undistort(img, mtx, dist, None, newcameramtx)
# dst = cv.resize(dst,(int(img.shape[1]/3),int(img.shape[0]/3)))
# # crop the image
# # x, y, w, h = roi
# # dst = dst[y:y+h, x:x+w]
# # cv.imwrite('calibresult.png', dst)
# cv.imshow("dst",dst)

# URL = '/home/ranger/Downloads/TEMP/'
# URL = '/media/ranger/01D454F10F8A7250/OneDrive - Institut Teknologi Bandung/CAMPUSS_S1/TAnjink/album/'
# URL = '/media/ranger/01D454F10F8A7250/OneDrive - Institut Teknologi Bandung/CAMPUSS_S1/TAnjink/'
# img = cv.imread(URL+'6Cp3Y.jpg')
# img = cv.imread(URL+'6e72267fb512e599dbf266807cb33aa6_preview_featured.jpg')
# img = cv.imread(URL+'lab/try1/camera_1561616230005.jpg') # percobaan skala lab
# camera_1561616084858
# camera_1561616124416 (copy)

# img = cv.imread(URL+'20190429_105518crop2.jpg') # percobaan skala lab
# img = cv.imread(URL+'Modular_Frame_system_for_paper_e.g._for_ArUco_markers_or_photos/images/29541ec755756d25fd405e8a622833bb_preview_featured.JPG')
# img = cv.imread(URL+'20190311_131316.jpg')

# aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
# aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_250)
aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
# aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
parameters = aruco.DetectorParameters_create()
# # cornerRefinement
WinSize = 15
minAcc = 0.04
maxIter = 42
# threshAruUpdate() init value
stepsizeHL = 10
minWin = 3
maxWin = 2  # maxWin adalah jumlah berapa step yang akan dilakukan
# cornerRefinement
# WinSize = 17
# minAcc = 0.04
# maxIter = 50
setattr(parameters, 'doCornerRefinement', 1)
# setattr(parameters, 'cornerRefinementMethod', int(aruco.CORNER_REFINE_SUBPIX))
# setattr(parameters, 'cornerRefinementMethod', 1)
setattr(parameters, 'cornerRefinementWinSize', WinSize)
setattr(parameters, 'cornerRefinementMinAccuracy', minAcc)
setattr(parameters, 'cornerRefinementMaxIterations', maxIter)
# threshAruUpdate -->
setattr(parameters, 'adaptiveThreshWinSizeMin', minWin)
setattr(parameters, 'adaptiveThreshWinSizeMax', int(minWin+WinSize*maxWin))
setattr(parameters, 'adaptiveThreshWinSizeStep', stepsizeHL)
# setattr(parameters,'polygonalApproxAccuracyRate',0.21)
'''/
images = glob.glob(URL+'kalibrasi/A6000/50mm-2/*.JPG')
print(datetime.datetime.now())
c = datetime.datetime.now()
for i in range(len(images)):
    print(i)
    img = cv.imread(images[i])
    img = cv.edgePreservingFilter(img, sigma_r=1)
    cv.imwrite(URL + 'kalibrasi/A6000/50mm-2/edgePreserving/' + str(i) + '.jpg', img)
    # cv.imwrite(URL+URLsample+'edgePreserving/'+str(i)+'.jpg',img)
images = glob.glob(URL+URLsample+'*.JPG')
for i in range(len(images)):
    print(i)
    img = cv.imread(images[i])
    img = cv.edgePreservingFilter(img, sigma_r=1)
    cv.imwrite(URL+URLsample+'edgePreserving/'+str(i)+'.jpg',img)
print(c)
print(datetime.datetime.now())
/'''
images = glob.glob(URL+URLsample+'*.JPG')
# cor_prior = np.zeros([4,2])
cor_prior = np.random.rand(4,2) # create dummy corners for first calculation
WP_prior = np.zeros([4,4])
WP_center_prior = np.zeros(3)
matr_prior = np.append(np.identity(3),[[0,0,0]],axis=0).transpose()
dist0 = np.zeros([1,5])
rvecp = np.array([[[1,1,1]]],dtype=np.float32)
tvecp = np.array([[[0,0,0]]],dtype=np.float32)
marker_size = 200 # length of marker side in mm
crop_margin = 50 # how much pixel as margin from the detected marker
c = datetime.datetime.now()
index = 0
images.sort()

for filename in images:
    index+=1
    stridxi = filename.index('DSC')
    stridxo = filename.index('.JP')
    db.write(filename[stridxi:stridxo]+',')
    print(filename)
    img = cv.imread(filename)
    img = cv.cvtColor(img,cv.COLOR_RGB2BGR)
    img = cv.undistort(img, mtx, dist, None, newcameramtx)
    # img = cv.edgePreservingFilter(img)
    f.write("\n\n"+filename+"\n")
    imag = img

    imrdx = img
    # img = cv.imread(URL+'shape_detection_thresh.jpg')
    imgt = img
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gery = gray
    gwery = gery
    # gwery = cv.resize(gery,(int(imgt.shape[1]/3),int(imgt.shape[0]/3)))
    # resized = imutils.resize(img, width=300)
    # ratio = img.shape[0] / float(resized.shape[0])
    title_window = 'thresholding'
    # img = cv.medianBlur(img,5)
    # cv.imshow('foto asli', img)
    gwerya = gwery
    # thresh = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #             cv.THRESH_BINARY,11,2)
    # blurred = cv.GaussianBlur(gray, (5, 5), 0)
    # thresh = cv.threshold(gray, 60, 255, cv.THRESH_BINARY)[1]
    # ret, thresh = cv.threshold(gray, 60, 255, 0)
    # blur = cv.GaussianBlur(gray,(5,5),0)
    # ret3,thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    # im2, contours, hierarchy = cv.findContours(threshx, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    # threshx = cv.resize(thresh,(int(imrdx.shape[1]/3),int(imrdx.shape[0]/3)))
    # cv.imshow(title_window,threshx)

    gery = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = gery

    # board = aruco.GridBoard_create(
    #         markersX=2,
    #         markersY=2,
    #         markerLength=0.09,
    #         markerSeparation=0.01,
    #         dictionary=aruco_dict)
    # print(parameters)

    # lists of ids and the corners beloning to each id
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    print(corners)
    if len(ids) > 1:
        img = aruco.drawDetectedMarkers(img, [corners[0]], borderColor=(0, 255, 0))
    else:
        img = aruco.drawDetectedMarkers(img, corners, borderColor=(0, 255, 0))

    '''/
    cv.namedWindow('cornerRefinementUpdate')

    def cornerRefinementUpdate(val):
        gray = gery
        img = imag
        setattr(parameters, 'cornerRefinementWinSize', int(2*cv.getTrackbarPos('WinSize','cornerRefinementUpdate')))
        setattr(parameters, 'cornerRefinementMinAccuracy', float(0.01*cv.getTrackbarPos('minAcc','cornerRefinementUpdate')))
        setattr(parameters, 'cornerRefinementMaxIterations', int(cv.getTrackbarPos('maxIter','cornerRefinementUpdate')))

        # threshAruUpdate -->
        # setattr(parameters, 'adaptiveThreshWinSizeMin', int(sminWin.val))
        # setattr(parameters, 'adaptiveThreshWinSizeMax', int(sminWin.val+sWinSize.val*smaxWin.val))
        # setattr(parameters, 'adaptiveThreshWinSizeStep', int(sstepsize.val))
        # print(sminWin.val+sstepsize.val*smaxWin.val)
        # lists of ids and the corners beloning to each id
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        # print(corners)

        # It's working.
        # my problem was that the cellphone put black all around it. The alrogithm
        # depends very much upon finding rectangular black blobs

        # gray = aruco.drawDetectedMarkers(gray, rejectedImgPoints)
        # img = aruco.drawDetectedMarkers(img, corners, borderColor=(0, 255, 0))
        # img = cv.resize(imag,(int(gray.shape[1]/3),int(gray.shape[0]/3)))
        # corners = [x/3 for x in corners]
        # rejectedImgPoints = [x/3 for x in rejectedImgPoints]
        rvec, tvec = aruco.estimatePoseSingleMarkers(corners, marker_size, mtx, dist)
        for i in range(0, len(rvec)):
            img = aruco.drawAxis(img, mtx, dist, rvec[i], tvec[i], marker_size)
        # img = aruco.drawDetectedMarkers(img, rejectedImgPoints, borderColor=(0, 0, 255))
        img = aruco.drawDetectedMarkers(img, corners, borderColor=(0, 255, 0))
        # img = aruco.drawDetectedMarkers(img, rejectedImgPoints, borderColor=(0, 0, 255))
        img = cv.resize(img, (int(gray.shape[1] / 4), int(gray.shape[0] / 4)))
        cv.imshow('cornerRefinementUpdate',img)
        # cv.waitKey()

    cv.createTrackbar('WinSize','cornerRefinementUpdate', 1, 25, cornerRefinementUpdate)
    cv.createTrackbar('minAcc', 'cornerRefinementUpdate', 1, 25, cornerRefinementUpdate)
    cv.createTrackbar('maxIter', 'cornerRefinementUpdate', 1, 100, cornerRefinementUpdate)
    img = cv.resize(img, (int(gray.shape[1] / 4), int(gray.shape[0] / 4)))
    cv.imshow('cornerRefinementUpdate',img)
    cv.waitKey()
    /'''

    '''    detectMarkers(...)
        detectMarkers(image, dictionary[, corners[, ids[, parameters[, rejectedI
        mgPoints]]]]) -> corners, ids, rejectedImgPoints
        '''
    # pyplot imshow
    '''/
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    edgePlot = ax1.imshow(img)
    plt.show()
    /'''
    # corner refinement update
    '''/
    # cornerRefinement slider
    axcolor = 'lightgoldenrodyellow'

    axWinSize = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
    axminAcc = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    axmaxIter = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)
    sWinSize = Slider(axWinSize, 'WinSize', 1, 25, valinit=WinSize, valstep=2)
    sminAcc = Slider(axminAcc, 'minAcc', 0.01, 0.2, valinit=minAcc, valstep=0.01)
    smaxIter = Slider(axmaxIter, 'maxIter', 1, 100, valinit=maxIter, valstep=1)

    # berikut threshAruUpdate()
    axstepsize = plt.axes([0.25, 0.3, 0.65, 0.03], facecolor=axcolor)
    axminWin = plt.axes([0.25, 0.25, 0.65, 0.03], facecolor=axcolor)
    axmaxWin = plt.axes([0.25, 0.2, 0.65, 0.03], facecolor=axcolor)
    sstepsize = Slider(axstepsize, 'stepsize', 0,20, valinit=stepsizeHL,valstep=2)
    sminWin = Slider(axminWin, 'minWin', 1,100, valinit=minWin,valstep=2)
    smaxWin = Slider(axmaxWin, 'maxWin', 1,100, valinit=maxWin,valstep=1)

    def cornerRefinementUpdate(val):
        img = cv.imread(filename)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # img = imag
        # gray = gery
        setattr(parameters, 'cornerRefinementWinSize', int(sWinSize.val))
        setattr(parameters, 'cornerRefinementMinAccuracy', float(sminAcc.val))
        setattr(parameters, 'cornerRefinementMaxIterations', int(smaxIter.val))

        # threshAruUpdate -->
        setattr(parameters, 'adaptiveThreshWinSizeMin', int(sminWin.val))
        setattr(parameters, 'adaptiveThreshWinSizeMax', int(sminWin.val+sWinSize.val*smaxWin.val))
        setattr(parameters, 'adaptiveThreshWinSizeStep', int(sstepsize.val))
        print(sminWin.val+sstepsize.val*smaxWin.val)
        # lists of ids and the corners beloning to each id
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        # print(corners)

        # It's working.
        # my problem was that the cellphone put black all around it. The alrogithm
        # depends very much upon finding rectangular black blobs

        # gray = aruco.drawDetectedMarkers(gray, rejectedImgPoints)
        # img = aruco.drawDetectedMarkers(img, corners, borderColor=(0, 255, 0))
        # img = cv.resize(imag,(int(gray.shape[1]/3),int(gray.shape[0]/3)))
        # corners = [x/3 for x in corners]
        # rejectedImgPoints = [x/3 for x in rejectedImgPoints]
        # rvec, tvec = aruco.estimatePoseSingleMarkers(corners, marker_size, mtx, dist)
        # for i in range(0, len(rvec)):
        #     img = aruco.drawAxis(img, mtx, dist, rvec[i], tvec[i], marker_size)
        # img = aruco.drawDetectedMarkers(img, rejectedImgPoints, borderColor=(0, 0, 255))
        imt = aruco.drawDetectedMarkers(img, corners, borderColor=(0, 255, 0))
        # img = aruco.drawDetectedMarkers(img, rejectedImgPoints, borderColor=(0, 0, 255))
        edgePlot.set_data(imt)
        fig1.canvas.draw_idle()
        # imt = cv.resize(img, (int(gray.shape[1] / 4), int(gray.shape[0] / 4)))
        # cv.imshow('cornerRefinementUpdate', imt)
        # cv.waitKey()

    sWinSize.on_changed(cornerRefinementUpdate)
    sminAcc.on_changed(cornerRefinementUpdate)
    smaxIter.on_changed(cornerRefinementUpdate)
    sstepsize.on_changed(cornerRefinementUpdate)
    sminWin.on_changed(cornerRefinementUpdate)
    smaxWin.on_changed(cornerRefinementUpdate)

    rax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
    radio = RadioButtons(rax, ('color', 'BLACK'), active=0)

    def blackSwitch(val):
        if val=='BLACK':
            edgePlot.set_data(imgblack)
            fig1.canvas.draw_idle()
        else:
            edgePlot.set_data(img)
            fig1.canvas.draw_idle()
    radio.on_clicked(blackSwitch)

    plt.show()

    /'''

    # '''/
    if len(ids)>1:
        rvec,tvec = aruco.estimatePoseSingleMarkers([corners[0]],marker_size,newcameramtx,dist0)
    else:
        rvec, tvec = aruco.estimatePoseSingleMarkers(corners, marker_size, newcameramtx, dist0)
    # rvec, tvec = aruco.estimatePoseSingleMarkers(corners, marker_size,mtx, dist)
    # tvecx = tvec.tolist()
    # for i in tvecx:
    #     for line in i:
    #         # f.write(" ".join(line) + "\n")
    #         f.write(str(line) + " ")

    print("corners:\n"+str(corners)+"\n------\n\n")
    f.write("corners:\n" + str(corners)+"\n")
    # f.write(tvec)
    # It's working.
    # my problem was that the cellphone put black all around it. The alrogithm
    # depends very much upon finding rectangular black blobs

    # gray = aruco.drawDetectedMarkers(gray, rejectedImgPoints)
    # img = aruco.drawDetectedMarkers(img, corners,borderColor=(0, 255, 0))
    # img = cv.resize(img,(int(gray.shape[1]/3),int(gray.shape[0]/3)))
    # corners = [x/3 for x in corners]
    # rejectedImgPoints = [x/3 for x in rejectedImgPoints]
    # '''/
    for i in range(0,len(rvec)):
        img = aruco.drawAxis(img,newcameramtx,dist0,rvec[i],tvec[i],marker_size)
        # img = aruco.drawAxis(img, mtx, dist, rvec[i], tvec[i], marker_size)
    # /'''
    # for corner in corners[0][0]:
    corpri = cor_prior
    cor_prior = np.insert(cor_prior, 2, 1, axis=1)
    cor = corners[0][0]
    cor = np.insert(cor,2,1,axis=1)

    a = (int(min(corners[0][0][0][1], corners[0][0][3][1])) - crop_margin)
    b = (int(max(corners[0][0][1][1], corners[0][0][2][1])) + crop_margin)
    c = int(min(corners[0][0][0][0], corners[0][0][1][0])) - crop_margin
    d = int(max(corners[0][0][2][0], corners[0][0][3][0])) + crop_margin
    roi = img[a:b,c:d]
    # roi = img[int(min(cor[0][0],cor[3][0]))-crop_margin:int(max(cor[1][0],cor[2][0]))+crop_margin, int(min(cor[0][1],cor[1][1]))-crop_margin:int(max(cor[2][1],cor[3][1]))+crop_margin]
    cv.imwrite(URL+URLsample+'detectedmarker/'+ filename[stridxi:],roi)
    rotmat, jacob = cv.Rodrigues(rvec)
    # wx = rvec[0][0][0]
    # wy = rvec[0][0][1]
    # wz = rvec[0][0][2]
    # tx = tvec[0][0][0]
    # ty = tvec[0][0][1]
    # tz = tvec[0][0][2]
    matr = np.append(rotmat.transpose(),tvec[0],axis=0).transpose()
    # matr = np.matmul(mtx,np.array([[1,-wz,wy,tx],[wz,1,-wx,ty],[-wy,wx,1,tz]]))
    WP = np.zeros([4,4])
    for i in range(len(cor)):
        WP_prior[i] = np.matmul(pinv(matr),cor_prior[i])
        WP[i] = np.matmul(pinv(matr),cor[i])
    print(WP.shape)
    print(WP)
    WP_center = np.zeros(3)
    ma = (WP[2][1]-WP[0][1])/(WP[2][0]-WP[0][0])
    mb = (WP[1][1]-WP[3][1])/(WP[1][0]-WP[3][0])
    WP_center[0] = (WP[3][1]-WP[0][1]+ma*WP[0][0]-mb*WP[3][0])/(ma-mb)
    WP_center[1] = ma*(WP_center[0]-WP[0][0])+WP[0][1]
    WP_center[2] = (WP[2][2]-WP[0][2])/(WP[2][0]-WP[0][0])*(WP_center[0]-WP[0][0])+WP[0][2]
    # WP_center[0] = (WP[0][1])/((WP[1][1]-WP[3][1])/(WP[1][0]-WP[3][0])-(WP[2][1]-WP[0][1])/(WP[2][0]-WP[0][0]))
    # WP_center[1] = (WP[1][1]-WP[3][1])/(WP[1][0]-WP[3][0])*WP_center[0]
    # WP_center[2] = (WP[1][2]-WP[3][2])/(WP[1][0]-WP[3][0])*WP_center[0]
    print(WP_center)

    ma = (WP_prior[2][1] - WP_prior[0][1]) / (WP_prior[2][0] - WP_prior[0][0])
    mb = (WP_prior[1][1] - WP_prior[3][1]) / (WP_prior[1][0] - WP_prior[3][0])
    WP_center_prior[0] = (WP_prior[3][1] - WP_prior[0][1] + ma * WP_prior[0][0] - mb * WP_prior[3][0]) / (ma - mb)
    WP_center_prior[1] = ma * (WP_center_prior[0] - WP_prior[0][0]) + WP_prior[0][1]
    WP_center_prior[2] = (WP_prior[2][2] - WP_prior[0][2]) / (WP_prior[2][0] - WP_prior[0][0]) * (WP_center_prior[0] - WP_prior[0][0]) + WP_prior[0][2]

    # WP_center_prior[0] = (WP_prior[0][1]) / ((WP_prior[1][1] - WP_prior[3][1]) / (WP_prior[1][0] - WP_prior[3][0]) - (WP_prior[2][1] - WP_prior[0][1]) / (WP_prior[2][0] - WP_prior[0][0]))
    # WP_center_prior[1] = (WP_prior[1][1] - WP_prior[3][1]) / (WP_prior[1][0] - WP_prior[3][0]) * WP_center_prior[0]
    # WP_center_prior[2] = (WP_prior[1][2] - WP_prior[3][2]) / (WP_prior[1][0] - WP_prior[3][0]) * WP_center_prior[0]
    f.write("\njarak:\n")
    # print(WP)
    jarak_rata = 0
    jarak_max = 0
    jarak_min = 1000
    for i in range(len(WP)):
        distX = abs(WP[i][0]-WP_prior[i][0])
        distY = abs(WP[i][1]-WP_prior[i][1])
        distZ = abs(WP[i][2]-WP_prior[i][2])
        jarake = math.sqrt(sum([(a - b) ** 2 for a, b in zip(WP[i], WP_prior[i])]))
        distYZ = math.sqrt(distY**2 + distZ**2)
        f.write("X = "+str(distX)+"; Z = "+str(distZ)+"; Y = "+str(distY)+"\ndist = "+str(jarake)+"\n"+"distYZ = "+str(distYZ)+"\n")
        # mengukur jarak selisih antara corner dan corner sebelumnya
        new = distance(cor[i], cor_prior[i], [1, 1, 0])
        jarak_min = min(jarak_min, new)
        jarak_max = max(jarak_max, new)
        jarak = new
        jarak_rata += new / 4
        f.write("dist corner dengan corner sebelumnya\ndist corner px" + str(i) + "= " + str(jarak))
    f.write("\njarak corner rata2 px = " + str(jarak_rata) + "mm/px jangkauan = " + str(jarak_max - jarak_min) + "px\n") #" kecermatan = " + str(marker_size / jarak_rata)
    db.write(str(jarak_min)+','+str(jarak_max)+','+str(jarak_rata)+',')
    distX = abs(WP_center[0] - WP_center_prior[0])
    distY = abs(WP_center[1] - WP_center_prior[1])
    distZ = abs(WP_center[2] - WP_center_prior[2])
    distYZctr = math.sqrt(distY ** 2 + distZ ** 2)
    f.write("center\nX = "+str(distX)+"; Z = "+str(distZ)+"; Y = "+str(distY)+"\ndistYZ center = " + str(distYZctr) + "\n")
    jarak_rata = 0
    jarak_max = 0
    jarak_min = 1000
    for i in range(4):
        new = distance(WP[i], WP[(i + 1) % 4], [1, 1, 1, 0])
        jarak_min = min(jarak_min,new)
        jarak_max = max(jarak_max,new)
        jarak = new
        jarak_rata += new/4
        f.write("\ndist corner WP"+str(i)+"-"+str((i+1)%4)+"= "+str(jarak))
    jangkauan = jarak_max-jarak_min
    jarak_total = distance(WP_center_prior, WP_center, [1, 1, 1, 0]) * marker_size / jarak_rata
    kecermatan = marker_size/jarak_rata
    f.write("\njarak corner rata2 WP = " + str(jarak_rata) +" kecermatan = " + str(kecermatan) +"mm/px jangkauan = " + str(jangkauan) +"px\nSetelah dinormalisasi, X= " + str(distance(WP_center_prior, WP_center, [1, 0, 0]) * marker_size / jarak_rata) + "; Y = " + str(distY * marker_size / jarak_rata) + "; distYZ center = " + str(distYZctr * marker_size / jarak_rata) + "; Z=" + str(distance(WP_center_prior, WP_center, [0, 0, 1, 0]) * marker_size / jarak_rata) + "jarak total= " + str(jarak_total) + "\n")
    db.write(str(jarak_min) + ',' + str(jarak_max) + ',' + str(jarak_rata) + ','+str(jangkauan)+','+str(kecermatan)+','+str(jarak_total))
    # mengukur jarak antar-corner dalam unit image plane
    jarak_rata = 0
    jarak_max = 0
    jarak_min = 1000
    for i in range(4):
        new = distance(corners[0][0][i], corners[0][0][(i + 1) % 4], [1, 1])
        jarak_min = min(jarak_min, new)
        jarak_max = max(jarak_max, new)
        jarak = new
        jarak_rata += new/4
        f.write("\ndist corner px" + str(i) + "-" + str((i + 1) % 4) + "= " + str(jarak))
    f.write("\njarak corner rata2 px = " + str(jarak_rata) + " kecermatan = " + str(
        marker_size / jarak_rata) + "mm/px jangkauan = " + str(
        jarak_max - jarak_min) + "px\n")

    # WP_prior = WP
    # WP_center_prior = WP_center
    WPp_RTp = cornerstoWP(corpri,rvecp,tvecp)
    WP_center_prior = WPofCenter(WPp_RTp)
    WP_RTp = cornerstoWP(corners[0][0], rvecp, tvecp)
    WP_center = WPofCenter(WP_RTp)
    distX = abs(WP_center[0] - WP_center_prior[0])
    distY = abs(WP_center[1] - WP_center_prior[1])
    distZ = abs(WP_center[2] - WP_center_prior[2])
    distYZctr = math.sqrt(distY ** 2 + distZ ** 2)
    f.write("RT old: center\nX = " + str(distX) + "; Z = " + str(distZ) + "; Y = " + str(distY) + "\ndistYZ center = " + str(
        distYZctr) + "\nSetelah dinormalisasi, Y = "+str(distY*marker_size/jarak_rata)+"; distYZ center = " + str(distYZctr*marker_size/jarak_rata) + "\n")
    f.write("\n-------\n\n")

    rvecp = rvec
    tvecp = tvec
    cor_prior = corners[0][0]
    db.write('\n')
    # /'''

    # img = aruco.drawDetectedMarkers(img, rejectedImgPoints,borderColor=(0, 0, 255))
    # print(rejectedImgPoints)
    # Display the resulting frame
    # cv.imshow('gray', img)
    # cv.waitKey()
    #
    # # When everything done, release the capture
    # cv.destroyAllWindows()
    # img = cv.resize(img,(int(img.shape[1]/3),int(img.shape[0]/3)))
    # cv.imshow("plot",img)
    # cv.waitKey()

    # fig1 = plt.figure()
    # ax1 = fig1.add_subplot(111)
    # edgePlot = ax1.imshow(img)
    # plt.show()
f.close()
db.close()
print(c)
print(datetime.datetime.now())
'''/
# axcolor = 'lightgoldenrodyellow'

# threshAruUpdate() init value
# stepsizeHL = 10
# minWin = 3
# maxWin = 2 # maxWin adalah jumlah berapa step yang akan dilakukan

# # berikut threshAruUpdate()
# axstepsize = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
# axminWin = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
# axmaxWin = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)
# sstepsize = Slider(axstepsize, 'stepsize', 0,20, valinit=stepsizeHL,valstep=2)
# sminWin = Slider(axminWin, 'minWin', 1,100, valinit=minWin,valstep=2)
# smaxWin = Slider(axmaxWin, 'maxWin', 1,100, valinit=maxWin,valstep=1)

# # cornerRefinement slider
# axWinSize = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
# axminAcc = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
# axmaxIter = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)
# sWinSize = Slider(axWinSize, 'WinSize', 1,20, valinit=WinSize,valstep=2)
# sminAcc = Slider(axminAcc, 'minAcc', 0.01,0.2, valinit=minAcc,valstep=0.01)
# smaxIter = Slider(axmaxIter, 'maxIter', 1,100, valinit=maxIter,valstep=1)

# def threshAruUpdate(val):
def cornerRefinementUpdate(val):
    gray = gery
    img = imag
    setattr(parameters, 'cornerRefinementWinSize', int(sWinSize.val))
    setattr(parameters, 'cornerRefinementMinAccuracy', float(sminAcc.val))
    setattr(parameters, 'cornerRefinementMaxIterations', int(smaxIter.val))

    # threshAruUpdate -->
    # setattr(parameters, 'adaptiveThreshWinSizeMin', int(sminWin.val))
    # setattr(parameters, 'adaptiveThreshWinSizeMax', int(sminWin.val+sWinSize.val*smaxWin.val))
    # setattr(parameters, 'adaptiveThreshWinSizeStep', int(sstepsize.val))
    # print(sminWin.val+sstepsize.val*smaxWin.val)
    # lists of ids and the corners beloning to each id
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    # print(corners)

    # It's working.
    # my problem was that the cellphone put black all around it. The alrogithm
    # depends very much upon finding rectangular black blobs

    # gray = aruco.drawDetectedMarkers(gray, rejectedImgPoints)
    # img = aruco.drawDetectedMarkers(img, corners, borderColor=(0, 255, 0))
    # img = cv.resize(imag,(int(gray.shape[1]/3),int(gray.shape[0]/3)))
    # corners = [x/3 for x in corners]
    # rejectedImgPoints = [x/3 for x in rejectedImgPoints]
    rvec, tvec = aruco.estimatePoseSingleMarkers(corners, marker_size, mtx, dist)
    for i in range(0, len(rvec)):
        img = aruco.drawAxis(img, mtx, dist, rvec[i], tvec[i], marker_size)
    # img = aruco.drawDetectedMarkers(img, rejectedImgPoints, borderColor=(0, 0, 255))
    img = aruco.drawDetectedMarkers(img, corners, borderColor=(0, 255, 0))
    # img = aruco.drawDetectedMarkers(img, rejectedImgPoints, borderColor=(0, 0, 255))
    edgePlot.set_data(img)
    fig1.canvas.draw_idle()
# sWinSize.on_changed(cornerRefinementUpdate)
# sminAcc.on_changed(cornerRefinementUpdate)
# smaxIter.on_changed(cornerRefinementUpdate)

plt.show()
/'''
print('selesai')