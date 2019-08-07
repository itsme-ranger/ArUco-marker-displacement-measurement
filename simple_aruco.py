from matplotlib import pyplot as plt
import cv2 as cv
import cv2.aruco as aruco
import numpy as np

URL = '/media/ranger/01D454F10F8A7250/OneDrive - Institut Teknologi Bandung/CAMPUSS_S1/TAnjink/'
URLsample = 'lab/try13/edgePreserving1/'
# URLsample = 'lab/try13/'
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
img1 = cv.imread(URL+URLsample+'DSC02678.JPG')
img1 = cv.cvtColor(img1,cv.COLOR_RGB2BGR)
gray = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)
corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
img1dr = aruco.drawDetectedMarkers(img1, corners, borderColor=(0, 255, 0))
plt.imshow(img1)
plt.figure()
'''/
crop_margin = 100
a= (int(min(corners[0][0][0][1],corners[0][0][3][1]))-crop_margin)
b= (int(max(corners[0][0][1][1],corners[0][0][2][1]))+crop_margin)
c= int(min(corners[0][0][0][0],corners[0][0][1][0]))-crop_margin
d= int(max(corners[0][0][2][0],corners[0][0][3][0]))+crop_margin
img1drcrop = img1dr[a:b,c:d]
plt.imshow(img1drcrop)
/'''
img1 = cv.imread(URL+URLsample+'DSC02677 step 0.5.JPG')
img1 = cv.cvtColor(img1,cv.COLOR_RGB2BGR)
gray = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)
corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
img1dr = aruco.drawDetectedMarkers(img1, corners, borderColor=(0, 255, 0))
plt.imshow(img1)
# plt.figure()
plt.show()
print corners[0][0]
print ('awaw')