# import poseEstimation as pe
# import calibration as clb
from __future__ import print_function
from __future__ import division
import argparse
import cv2 as cv
import shapedetector
import math
import numpy as np
from matplotlib.widgets import Slider
from matplotlib import pyplot as plt
import imutils
import operator

# matrixCam = clb.distorsi(chessboard, lebar, tinggi) #input matrix intrinsik kamera
# input model / Read 3D textured object model and object mesh.
# input video / cam
# Match scene descriptors with model descriptors using Flann matcher
# descriptor 1: pake nearest neighbor
# descriptor 2: pake corner dalam boundary

'''/
def thresholding(val):
    imgt = img
    # khusus untuk ADAPTIVA GAUSSIAN
    if val < 3:
        val = 11
    else:
        val = val * 2 + 1
    # thresholding
    # ret, threshx = cv.threshold(gray, val, 255, 0)
    # threshx = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\cv.THRESH_BINARY,val,2)

    plt.imshow(threshx,'gray')
    plt.show()
    
    threshx = np.float32(threshx)
    dst = cv.cornerHarris(threshx, 2, 3, 0.04)  # CV_SCHARR (-1)
    # result is dilated for marking the corners, not important
    dst = cv.dilate(dst, None)
    # Threshold for an optimal value, it may vary depending on the image.
    # cv.imshow(title_window, img)
    imgt[dst > 0.01 * dst.max()] = [0, 0, 255]
    cv.imshow(title_window, imgt)
/'''

# URL = '/home/ranger/Downloads/TEMP/'
URL = '/media/ranger/01D454F10F8A7250/OneDrive - Institut Teknologi Bandung/CAMPUSS_S1/TAnjink/album/'
# URL = '/media/ranger/01D454F10F8A7250/OneDrive - Institut Teknologi Bandung/CAMPUSS_S1/TAnjink/'
# img = cv.imread(URL+'line_102681123776169.jpg')
# img = cv.imread(URL+'line_102680503331516.jpg') # zoom
# img = cv.imread(URL+'bentuk sembarang.jpg')
# img = cv.imread(URL+'20190311_131414.jpg')
# img = cv.imread(URL+'20190311_131338.jpg')
img = cv.imread(URL+'20190311_131310.jpg') # 2 kotak hitam putiih
# img = cv.imread(URL+'20190311_131316.jpg')
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
'''/
alpha = 3
beta = 0

fig3 = plt.figure()
ax31 = fig3.add_subplot(111)
for y in range(gwery.shape[0]):
    for x in range(gwery.shape[1]):
        gwerya[y, x] = np.clip(alpha * gwery[y, x] + beta, 0, 255)
# ax32 = fig3.add_subplot(122)
# imPlot = ax31.imshow(gwerya)
cv.imshow('contrast',gwerya)
cv.waitKey()
/'''
# histPlot = ax32.hist(img.ravel(),256,[0,256])
'''/
axcolor = 'lightgoldenrodyellow'
axalpha = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
axbeta = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
salpha = Slider(axalpha, 'contrast', 1,3, valinit=alpha,valstep=0.1)
sbeta = Slider(axbeta, 'brightness', 1,100, valinit=beta,valstep=1)
def histUpdate(val):

    alpha = salpha.val
    beta = sbeta.val
    for y in range(gwery.shape[0]):
        for x in range(gwery.shape[1]):
            gwerya[y, x] = np.clip(alpha * gwery[y, x] + beta, 0, 255)
    # l.set_ydata(amp*np.sin(2*np.pi*freq*t))
    imPlot.set_data(gwerya)
    # histPlot.set_data(plt.hist(gwery.ravel(),256,[0,256]))
    fig3.canvas.draw_idle()
salpha.on_changed(histUpdate)
sbeta.on_changed(histUpdate)
plt.show()
/'''

# thresh = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
#             cv.THRESH_BINARY,11,2)
# blurred = cv.GaussianBlur(gray, (5, 5), 0)
# thresh = cv.threshold(gray, 60, 255, cv.THRESH_BINARY)[1]
# ret, thresh = cv.threshold(gray, 60, 255, 0)
# blur = cv.GaussianBlur(gray,(5,5),0)
ret3,thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
# im2, contours, hierarchy = cv.findContours(threshx, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
dxxthreshx = cv.resize(thresh,(int(imrdx.shape[1]/3),int(imrdx.shape[0]/3)))
cv.imshow(title_window,threshx)
'''/
for i in range(len(contours)):
    for j in range(len(contours[i])):
        imgt[contours[i][j][0][0]][contours[i][j][0][1]] = [0, 255, 0]
/'''
# cv.imshow('thres',threshx)
# cv.imshow('udah',imgt)

minVal = 23 #130
maxVal = 51 #135
edges = cv.Canny(gery, minVal, maxVal)
# edgesx = cv.resize(edges,(int(imrdx.shape[1]/3),int(imrdx.shape[0]/3)))
# cv.imshow("cannyy",edgesx)
'''/
cv.imshow('canny',edges)
cv.waitKey()
/'''
'''/
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)

edgePlot = ax1.imshow(edges, cmap='gray')
axcolor = 'lightgoldenrodyellow'

axminVal = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
axmaxVal = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
sminVal = Slider(axminVal, 'minVal', 1,255, valinit=minVal,valstep=1)
smaxVal = Slider(axmaxVal, 'maxVal', 1,255, valinit=maxVal,valstep=1)

def cannyUpdate(val):
    gerey = gray
    minVal = sminVal.val
    maxVal = smaxVal.val
    # l.set_ydata(amp*np.sin(2*np.pi*freq*t))
    edgeu = cv.Canny(gerey, minVal, maxVal)
    edgePlot.set_data(edgeu)
    # edgePlot.set_data(gery)
    fig1.canvas.draw_idle()
    return edgeu
sminVal.on_changed(cannyUpdate)
smaxVal.on_changed(cannyUpdate)

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)

sret = [[255 if x == 0 else 0 for x in w] for w in thresh]
sret = np.array(sret)
# sret = cv.resize(sret,(int(gery.shape[1]/3),int(gery.shape[0]/3)))
# ax2.imshow(sret,cmap='gray')
# cv.imshow('sret',sret)
# cv.waitKey()
# plt.show()
linesP = cv.HoughLinesP(sret, 1, np.pi / 180, 50, None, 50, 10)
if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(sret, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)
sret = cv.resize(sret,(int(gery.shape[1]/3),int(gery.shape[0]/3)))
# houghPlot = ax2.imshow(gery)
cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", gery)
/'''
threshHL = 50
minLinLength = 50
maxLineGap = 10

axthresh = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
axminLinLength = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
axmaxLineGap = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)
sthresh = Slider(axthresh, 'thresh', 1,100, valinit=threshHL,valstep=1)
sminLinLength = Slider(axminLinLength, 'minlinlength', 1,100, valinit=minLinLength,valstep=1)
smaxLineGap = Slider(axmaxLineGap, 'maxLineGap', 1,100, valinit=maxLineGap,valstep=1)

def houghLinesUpdate(val):
    # l.set_ydata(amp*np.sin(2*np.pi*freq*t))
    edgeu = cannyUpdate()
    linesP = cv.HoughLinesP(edgeu, 1, np.pi / 180, sthresh.val, None, sminLinLength.val, smaxLineGap.val)
    gery = gray
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(gery, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)

    houghPlot.set_data(gery)
    fig2.canvas.draw_idle()
# sminVal.on_changed(houghLinesUpdate)
# smaxVal.on_changed(houghLinesUpdate)
sthresh.on_changed(houghLinesUpdate)
sminLinLength.on_changed(houghLinesUpdate)
smaxLineGap.on_changed(houghLinesUpdate)

'''/
linesP = cv.HoughLinesP(edges, 1, np.pi / 180, 50, None, 50, 10)
if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(imgt, (l[0], l[1]), (l[2], l[3]), (255,0,0), 3, cv.LINE_AA)
edgePlot = ax1.imshow(imgt)
/'''

# shape detection
epsHL = 4
# axeps = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
# seps = Slider(axeps, 'eps', 0,30, valinit=epsHL,valstep=1)

# cnts = cv.findContours(thresh, cv.RETR_CCOMP,cv.CHAIN_APPROX_NONE)
cnts = cv.findContours(thresh, cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
# cv.drawContours(imgt, cnts, -1, (0,255,0), 3)
# cv.imshow('kontor',imgt)
sd = shapedetector.ShapeDetector()

# deklarasi luas marker dalam persen
markerAreaWmin = 10
markerAreaWmax = 35
markerAreaHmin = 10
markerAreaHmax = 80

# loop over the contours
dck = sorted(cnts, key=cv.contourArea, reverse=True)
for c in dck:
    # compute the center of the contour, then detect the name of the
    # shape using only the contour
    # print(c)
    M = cv.moments(c)
    # cX = int((M["m10"] / M["m00"]) * ratio)
    # cY = int((M["m01"] / M["m00"]) * ratio)
    [shape, appr] = sd.detect(c,epsHL)

    '''/
    c = c.astype("float")
    c *= ratio
    c = c.astype("int")
    cv.drawContours(imrdx, [c], 0, (0, 255, 0), 3)
    print('draw done')
    /'''
    if (shape == 'square') or (shape == 'rectangle'):
        amin = markerAreaHmin * markerAreaWmin * img.shape[0] * img.shape[1] / (10 ** 4)
        amax = markerAreaHmax*markerAreaWmax*img.shape[0]*img.shape[1]/(10**4)
        print(cv.contourArea(c))

        if (cv.contourArea(c) > amax):
            continue
        if (cv.contourArea(c) < amin):
            break

        # multiply the contour (x, y)-coordinates by the resize ratio,
        # then draw the contours and the name of the shape on the image
        '''/
        for b in c:
            imrdx[b[0][1]][b[0][0]] = [0, 0, 255]
            /'''
        # c = c.astype("float")
        # c *= ratio
        # c = c.astype("int")
        cv.drawContours(imrdx, [c], -1, (0, 255, 0), 4)
        [cv.line(imrdx,(appr[i][0][0],appr[i][0][1]),(appr[(i+1)%4][0][0],appr[(i+1)%4][0][1]),(255,0,0),2,cv.LINE_AA) for i in range(4)]
        print('draw done')
        # cv.putText(imrdp, cv.contourArea(c), (cX, cY), cv.FONT_HERSHEY_SIMPLEX,6, (255, 0, 0), 3)
# show the output image
# imrdpPlot = ax2.imshow(imrdx)
cv.imwrite(URL+'keluar2.png',imrdx)
cv.namedWindow('gam',cv.WINDOW_NORMAL)
imrdxs = cv.resize(imrdx,(int(imrdx.shape[1]/3),int(imrdx.shape[0]/3)))
# cv.resizeWindow('gam',imrdxs)
cv.imshow('gam',imrdxs)
# Ramer-Douglas-Peucker algorithm
'''/
def RDP(val):
    imrdp = img
    for c in cnts:
        # compute the center of the contour, then detect the name of the
        # shape using only the contour
        M = cv.moments(c)
        # cX = int((M["m10"] / M["m00"]) * ratio)
        # cY = int((M["m01"] / M["m00"]) * ratio)
        shape = sd.detect(c, seps.val)

        # multiply the contour (x, y)-coordinates by the resize ratio,
        # then draw the contours and the name of the shape on the image
        c = c.astype("float")
        c *= ratio
        c = c.astype("int")
        cv.drawContours(imrdp, [c], -1, (0, 255, 0), 2)
        # cv.putText(imrdp, shape, (cX, cY), cv.FONT_HERSHEY_SIMPLEX,6, (255, 0, 0), 3)

    # show the output image
    imrdpPlot.set_data(imrdp)
    fig2.canvas.draw_idle()
seps.on_changed(RDP)
/'''
plt.xticks([]),plt.yticks([])
plt.show()
cv.waitKey()

# line detection
# lines = cv.HoughLinesP(edges, 1, np.pi / 180, 50, None, 50, 10)
cv.waitKey()
'''/
if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        n
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        cv.line(gery, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)
/'''
'''/
cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", gery)
cv.waitKey()
threshx = np.float32(threshx)
dst = cv.cornerHarris(threshx, 2, 3, 0.04)  # CV_SCHARR (-1)
# result is dilated for marking the corners, not important
dst = cv.dilate(dst, None)
# Threshold for an optimal value, it may vary depending on the image.
# cv.imshow(title_window, img)
imgt[dst > 0.01 * dst.max()] = [0, 0, 255]
cv.imshow(title_window, imgt)
/'''
'''/
ret, threshx = cv.threshold(gray, 127, 255, 0)

threshx = np.float32(threshx)
dst = cv.cornerHarris(threshx,2,3,0.04) # CV_SCHARR (-1)
#result is dilated for marking the corners, not important
dst = cv.dilate(dst,None)
# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]
cv.imshow(title_window,img)
/'''

# nunjukin slider
'''/
ksizeMax = 6
bsizeMax = 6
kMax = 0.07

cv.namedWindow(title_window)
trackbar_name = 'Alpha x %d' % 255
cv.createTrackbar(trackbar_name, title_window , 1, 7, thresholding)
# Show some stuff
thresholding(0)
/'''
# Wait until user press some key
# cv.waitKey()



if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()


# pose estimaation