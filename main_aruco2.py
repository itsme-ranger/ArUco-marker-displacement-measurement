from __future__ import print_function
from __future__ import division
from numpy.linalg import pinv, inv,norm
import argparse
import cv2 as cv
import cv2.aruco as aruco
# import shapedetector
import math
import numpy as np
from matplotlib.widgets import Slider, RadioButtons
from matplotlib import pyplot as plt
import imutils
import operator
import glob
import datetime


def swap(a, b):
    return b, a


def cornerstoWP(corners, mtx, rvec, tvec):
    # cor = corners[0][0]
    corners = np.insert(corners, 2, 1, axis=1)
    rotmat, jacob = cv.Rodrigues(rvec)
    matr = np.append(rotmat.transpose(), tvec, axis=0).transpose()
    print(rvec, tvec)
    print("matr = ")
    print(matr)
    WP = np.zeros([4, 4])
    for i in range(len(corners)):
        # mat = np.matmul(inv(mtx),corners[i])
        # WP[i] = np.matmul(pinv(matr),mat)
        mat = np.matmul(pinv(matr), inv(mtx))
        WP[i] = np.matmul(mat, corners[i])
    return WP


def WPtoreal(WP, length):
    jarak_rata = 0
    jarak_max = 0
    jarak_min = 100000
    for i in range(4):
        new = distance(WP[i], WP[(i + 1) % 4], [1, 1, 1, 0])
        print(new)
        jarak_min = min(jarak_min, new)
        jarak_max = max(jarak_max, new)
        jarak = new
        jarak_rata += new / 4
    return length / jarak_rata


def WPofCenter(WP):
    WP_center = np.zeros(3)
    ma = (WP[2][1] - WP[0][1]) / (WP[2][0] - WP[0][0])
    mb = (WP[1][1] - WP[3][1]) / (WP[1][0] - WP[3][0])
    WP_center[0] = (WP[3][1] - WP[0][1] + ma * WP[0][0] - mb * WP[3][0]) / (ma - mb)
    WP_center[1] = ma * (WP_center[0] - WP[0][0]) + WP[0][1]
    WP_center[2] = (WP[2][2] - WP[0][2]) / (WP[2][0] - WP[0][0]) * (WP_center[0] - WP[0][0]) + WP[0][2]
    return WP_center


def distance(start, target, mat):
    sum = 0
    for x in mat:
        sum += x
    distnce = 0
    if sum != 1:
        for i in range(len(start)):
            distnce += mat[i] * (target[i] - start[i]) ** 2
        return math.sqrt(distnce)
    else:
        for i in range(len(start)):
            distnce += mat[i] * (target[i] - start[i])
        return distnce

ep = 'edgePreserving/'
st = 'stabilized/'
URL = '/media/ranger/01D454F10F8A7250/OneDrive - Institut Teknologi Bandung/CAMPUSS_S1/TAnjink/'
URLsample = 'lab/try15/edgePreserving/'
# URL = 'E:/OneDrive - Institut Teknologi Bandung/CAMPUSS_S1/TAnjink/'
# URL = '/home/ranger/Documents/Project/'
URLsample = 'lab/try17/'+st+ep
# URLsample = 'lab/try15/'
# URLsample = 'lab/try13/'
URLstrip = ['-' if x == '/' else x for x in URLsample]
URLstrip = ''.join(URLstrip)
now = datetime.datetime.now()
# f = open("tabel translasi.txt","a+")
# f = open("sampah.txt","a+")
f = open(URL + URLsample + "invoice " + URLstrip + ' ' + str(now.strftime("%m-%d_%H-%M")) + ".txt", "a+")
db = open(URL + URLsample + "db " + URLstrip + ' ' + str(now.strftime("%m-%d_%H-%M")) + ".csv", "a+")
errorxyz = open(URL + URLsample + "error xyz " + URLstrip + ' ' + str(now.strftime("%m-%d_%H-%M")) + ".csv", "a+")
errorside = open(URL + URLsample + "error side " + URLstrip + ' ' + str(now.strftime("%m-%d_%H-%M")) + ".csv", "a+")
# errorcenter = open("error center "+URLstrip+' '+str(now.strftime("%m-%d_%H-%M"))+".csv","a+")
# db = open("db.csv","a+")
header = '\n\n' + str(URLsample) + '\n' + str(datetime.datetime.now()) + '\n'
db.write(header)
errorside.write(header)
errorxyz.write(header)
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
dist = [[9.53306186e-03, -5.61930848e-02, -8.66880870e-03, -1.60162599e-03, -5.23222839e+00]]
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
print(newcameramtx)
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
# aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
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
# setattr(parameters, 'doCornerRefinement', 1)
# setattr(parameters, 'cornerRefinementMethod', int(aruco.CORNER_REFINE_SUBPIX))
# setattr(parameters, 'cornerRefinementMethod', 1)
setattr(parameters, 'cornerRefinementWinSize', WinSize)
setattr(parameters, 'cornerRefinementMinAccuracy', minAcc)
setattr(parameters, 'cornerRefinementMaxIterations', maxIter)
# threshAruUpdate -->
setattr(parameters, 'adaptiveThreshWinSizeMin', minWin)
setattr(parameters, 'adaptiveThreshWinSizeMax', int(minWin + WinSize * maxWin))
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
images = glob.glob(URL + URLsample + '*.JPG')
# cor_prior = np.zeros([4,2])
cor_prior = np.random.rand(4, 2)  # create dummy corners for first calculation
corpri = cor_prior
WP_prior = np.zeros([4, 4])
WP_center_prior = np.zeros(3)
matr_prior = np.append(np.identity(3), [[0, 0, 0]], axis=0).transpose()
dist0 = np.zeros([1, 5])
rvecp = np.array([[[1, 1, 1]]], dtype=np.float32)
tvecp = np.array([[[0, 0, 0]]], dtype=np.float32)
crop_margin = 50  # how much pixel as margin from the detected marker
c = datetime.datetime.now()
avec = np.zeros([2, 3])
bvec = np.zeros([2, 3])
markers_id = [14,13]
marker_size = { # length of marker side in mm
    markers_id[0]:140,
    markers_id[1]:105}
# markers_id = [30]
index = 0
images.sort()
# header1 = 'jarak min,jarak max,jarak rata,jarak min,jarak max,jarak rata,jangkauan,kecermatan,jarak total,'
header1 = ''
header2 = 'crnr 0,crnr 1,crnr 2,crnr 3,'
header3 = 'Jarak referensi,jarak ST3,kecermatan (mm/px)'
db.write('nama file,'+header1+header2+header3+'\n')
newcameramtx,dist0 = mtx,dist
for filename in images:
    index += 1
    stridxi = filename.index('DSC')
    stridxo = filename.index('.JP')
    db.write(filename[stridxi:stridxo])
    print(filename)
    img = cv.imread(filename)
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    # img = cv.undistort(img, mtx, dist, None, newcameramtx)
    # img = cv.edgePreservingFilter(img)
    f.write("\n\n" + filename + "\n")
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
    # STAGE 2
    # MEASUREMENT BASED ON DISTANCE RELATIVE FROM MARKER REFERENCE
    cornersdict, rvecsdict, tvecsdict = [{}, {}, {}]
    # cornersnew = []
    f.write("\nid = ")
    i, ref, tg = 0, markers_id[0], markers_id[1]
    note = [[-1, 1, 1, -1], [1, 1, -1, -1], [0, 0, 0, 0]]
    for x in range(len(corners)):
        if ids[x][0] in markers_id:
            f.write(str(ids[x][0]) + "\n")
            cornersdict[ids[x][0]] = corners[x][0]
            f.write(str(corners[x][0])+'\n')
            rvecs, tvecs = aruco.estimatePoseSingleMarkers([corners[x]], marker_size[ids[x][0]], newcameramtx, dist0)
            rvecsdict[ids[x][0]] = rvecs[0]
            tvecsdict[ids[x][0]] = tvecs[0]
            # cornersnew.append(corners[x])
            # if (ids[x][0] == 12) & (i!=ref):
            # avec = swap(avec[0],avec[1])
            # ref = (ref+1)%2
            i += 1
    # cornersnew = np.asarray(cornersnew)
    # '''/
    resolution = 0
    for k in range(4):
        resolution += norm(cornersdict[tg][(k+1)%4]-cornersdict[tg][k])
        db.write(','+str(norm(cornersdict[tg][k]-corpri[k])))
    resolution = marker_size[tg]/resolution*4

    j = 0
    if ref in cornersdict:
        WP_ref = cornerstoWP(cornersdict[ref], newcameramtx, rvecsdict[tg], tvecsdict[tg])
        WP = cornerstoWP(cornersdict[tg], newcameramtx, rvecsdict[tg], tvecsdict[tg])
        resolutionWP = WPtoreal(WP, marker_size[tg])
        f.write("\njarak antar corner dalam WP target = \n")
        for i in range(4):
            WP_ref[i] = WP_ref[i] / WP_ref[i][3]
            WP[i] = WP[i] / WP[i][3]
        ka = np.zeros(4)
        for i in range(4):
            print(WP[i][0:3])
            jarak = norm(WP[i][0:3] - WP[(i + 1) % 4][0:3])
            f.write(str(jarak) + " ")
            for k in range(3):
                errorxyz.write(str(WP[i][k] - marker_size[tg]/ 2 * note[k][i]) + ",")
            errorxyz.write("\n")
            errorside.write(str(jarak - marker_size[tg]) + "\n")
            ka[i] = norm(WP[i][0:3] - WP_ref[i][0:3])
            f.write("ka[i] " + str(ka[i]) + " ")
        center_ref = WPofCenter(WP_ref)
        center = WPofCenter(WP)
        for i in range(3):
            bvec[j][i] = (center[i] - center_ref[i]) * resolutionWP
        print(avec[j], bvec[j])

        f.write("\nR|T target, WP = \n" + str(WP) + " WP ref = \n" + str(WP_ref) + "\njarak antar corner WP ref = \n")
        WP_ref = cornerstoWP(cornersdict[ref], newcameramtx, rvecsdict[ref], tvecsdict[ref])
        WP = cornerstoWP(cornersdict[tg], newcameramtx, rvecsdict[ref], tvecsdict[ref])
        for i in range(4):
            WP_ref[i] = WP_ref[i] / WP_ref[i][3]
            WP[i] = WP[i] / WP[i][3]
        kb = np.zeros(4)
        for i in range(4):
            jarak = norm(WP_ref[(i + 1) % 4][0:3] - WP_ref[i][0:3])
            f.write(str(jarak) + " ")
            kb[i] = norm(WP[i][0:3] - WP_ref[i][0:3])
            f.write("kb[i] " + str(kb[i]) + " ")
            f.write(str(kb[i] - ka[i]) + " ")
            for k in range(3):
                errorxyz.write(str(WP_ref[i][k] - marker_size[ref] / 2 * note[k][i]) + ",")
            errorxyz.write("\n")
            errorside.write(str(jarak - marker_size[ref]) + "\n")
        f.write("\nR|T referen, WP = \n" + str(WP) + " WP ref = \n" + str(WP_ref))

        jarak = norm(avec[j]) ** 2 + norm(bvec[j]) ** 2 - 2 * np.dot(avec[j], bvec[j])
        if jarak < 0:
            jarak = 0
        print(norm(avec[j]) ** 2, norm(bvec[j]) ** 2, 2 * np.dot(avec[j], bvec[j]))
        print(jarak)
        # errorcenter.write(j)
        jarak = math.sqrt(jarak)

        print(jarak)
        db.write(',' + str(jarak))
        f.write("\njarak pakai referensi = " + str(jarak) + "\n-------\n\n")
        avec[j] = tuple(bvec[j])
    # /'''
    db.write(',')
    j = 1
    # STAGE 3
    # ACTUALLY JUST STAGE 1 WITH MORE ELEGANT CODEBASE
    WP = cornerstoWP(cornersdict[tg], newcameramtx, rvecsdict[tg], tvecsdict[tg])
    WP_ref = cornerstoWP(corpri, newcameramtx, rvecsdict[tg], tvecsdict[tg])
    resolutionWP = WPtoreal(WP, marker_size[tg])
    f.write("\njarak antar corner dalam WP target = \n")
    for i in range(4):
        WP_ref[i] = WP_ref[i] / WP_ref[i][3]
        WP[i] = WP[i] / WP[i][3]
    # ka=np.zeros(4)
    for i in range(4):
        print(WP[i][0:3])
        jarak = norm(WP[i][0:3] - WP[(i + 1) % 4][0:3])
        f.write(str(jarak) + " ")
        for k in range(3):
            errorxyz.write(str(WP[i][k] - marker_size[tg] / 2 * note[k][i]) + ",")
        errorxyz.write("\n")
        errorside.write(str(jarak - marker_size[tg]) + "\n")
        # ka[i] = norm(WP[i][0:3]-WP_ref[i][0:3])
        # f.write("ka[i] "+str(ka[i])+" ")
    center_ref = WPofCenter(WP_ref)
    center = WPofCenter(WP)
    for i in range(3):
        bvec[j][i] = (center[i] - center_ref[i]) * resolutionWP
    jarak = math.sqrt(norm(bvec[j]))
    if jarak < 0:
        jarak = 0
    db.write(',' + str(jarak)+','+str(resolution))

    avec[j] = tuple(bvec[j])
    # rvecp = rvec
    # tvecp = tvec
    cor_prior = corners[0][0]
    corpri = cornersdict[tg]
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
errorxyz.close()
errorside.close()
print(c)
print(datetime.datetime.now())
print('selesai')