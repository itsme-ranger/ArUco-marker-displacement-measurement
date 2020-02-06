from matplotlib import pyplot as plt
import cv2 as cv
import cv2.aruco as aruco
import numpy as np

ep = 'edgePreserving/'
URL = '/media/ranger/01D454F10F8A7250/OneDrive - Institut Teknologi Bandung/CAMPUSS_S1/TAnjink/'
URLsample = 'lab/try13/edgePreserving1/'
URL = '/home/ranger/Documents/Project/'
URLsample = 'try17/'
# URLsample = 'lab/try13/'
URLstrip = ['-' if x == '/' else x for x in URLsample]
URLstrip = ''.join(URLstrip)
# now = datetime.datetime.now()
# f = open("tabel translasi.txt","a+")
# f = open("sampah.txt","a+")
# f = open("invoice "+URLstrip+' '+str(now.strftime("%m-%d_%H-%M"))+".txt","a+")

# aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
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
'''/
img1 = cv.imread(URL+URLsample+'DSC02678.JPG')
img1 = cv.cvtColor(img1,cv.COLOR_RGB2BGR)
gray = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)
corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
img1dr = aruco.drawDetectedMarkers(img1, corners, borderColor=(0, 255, 0))
plt.imshow(img1)
plt.figure()
/'''
'''/
crop_margin = 100
a= (int(min(corners[0][0][0][1],corners[0][0][3][1]))-crop_margin)
b= (int(max(corners[0][0][1][1],corners[0][0][2][1]))+crop_margin)
c= int(min(corners[0][0][0][0],corners[0][0][1][0]))-crop_margin
d= int(max(corners[0][0][2][0],corners[0][0][3][0]))+crop_margin
img1drcrop = img1dr[a:b,c:d]
plt.imshow(img1drcrop)
/'''
# imfile = 'DSC02858'
imfile = 'DSC02884'
ext = 'JPG'
newcameramtx = [[1.30038496e+04, 0.00000000e+00, 2.94683246e+03],
                [0.00000000e+00, 1.30000098e+04, 1.67097140e+03],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
newcameramtx = np.asarray(newcameramtx)
dist0 = np.zeros([1, 5])
marker_size = 105

img1 = cv.imread(URL+URLsample+imfile+'.'+ext)
img1 = cv.cvtColor(img1,cv.COLOR_RGB2BGR)
gray = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)
corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
rvecs, tvecs = aruco.estimatePoseSingleMarkers(corners, marker_size, newcameramtx, dist0)
img1dr = aruco.drawDetectedMarkers(img1, corners, borderColor=(0, 255, 0))
plt.imshow(img1dr)
# plt.figure()
plt.show()
print corners[0][0]
print ('awaw')