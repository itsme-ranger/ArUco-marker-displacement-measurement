import cv2 as cv
import cv2.aruco as aruco
import math
import numpy as np
from .config import *

def filename2displacement(filename,extension,separator):
    if separator not in filename:
        return 0
    else
        stridxi = filename.index(separator)
        stridxo = filename.index('.'+extension)
        return float(filename[stridxi+len(separator):stridxo])

def swap(a, b):
    return b, a

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


class Marker(object):
    """docstring for Marker"""
    def __init__(self, _id, size):
        super(Marker, self).__init__()
        # self.arg = arg
        self._id = _id
        self.marker_size = size
        self.idx_displace = 1
        self.idx = {} # dict of filename as keys and their file index number as value
        self.cornersArr = [] # array of marker's corners
        self.cornersArr.append(np.random.rand(4, 2))
        self.displacement_actual = [] # displacement you give to the system
        self.displacement_measured = [] # displacement measured from the camera
        self.displacement_averaged = [] # displacement measured from the camera
        self.error_side_Var = []

    def cornerstoWP(self,corners):
        rvec, tvec = aruco.estimatePoseSingleMarkers([corners], self.marker_size, mtx, dist)
        corners = np.insert(corners, 2, 1, axis=1)
        rotmat, jacob = cv.Rodrigues(rvec)
        matr = np.append(rotmat.transpose(), tvec, axis=0).transpose()
        
        WP = np.zeros([4, 4])
        for i in range(len(corners)):
            mat = np.matmul(pinv(matr), inv(mtx))
            WP[i] = np.matmul(mat, corners[i])
        for i in range(4):
            WP[i] = WP[i] / WP[i][3]
        return WP

    def measure(self,corners_prior,corners_now):
        WP = cornerstoWP(corners_now)
        WP_prior = cornerstoWP(corners_prior)
        resolutionWP = WPtoreal(WP, marker_size[self._id])
        center_prior = WPofCenter(WP_prior)
        center = WPofCenter(WP)
        displace = np.zeros(3)
        for i in range(3):
            displace[i] = (center[i] - center_prior[i]) * resolutionWP
        return math.sqrt(norm(displace))

    def error_side_calc(self,corners):
        """ Calculate the differences of length between measured side in World Point and actual length """
        # note = [[-1, 1, 1, -1], [1, 1, -1, -1], [0, 0, 0, 0]]
        position = ['up','right','down','left']
        error_side_dict = dict.fromkeys(position)
        WP = cornerstoWP(corners)

        for i,(k,v) in enumerate(error_side_dict.items()):
            distance = norm(WP[i][0:3] - WP[(i + 1) % 4][0:3])
            error_side_dict[k] = distance - self.marker_size
        return error_side_dict

    def __call__(self,corners,file_address):
        filename = file_address[file_address.rfind('/'):file_address.find('.'+extension)]
        self.idx[filename] = len(cornersArr)-1
        self.cornersArr.append(corners)
        displacement_actual = filename2displacement(filename,extension,separator)
        self.displacement_actual.append(displacement_actual)
        self.displacement_measured.append(self.measure(cornersArr[-2],corners))
        
        if displacement_actual != 0.0:
            corners_avg = self.displacement_measured[self.idx_displace:len(displacement_measured)-2].mean(axis=0)
            self.displacement_averaged.append(measure(corners_avg,corners))
            self.idx_displace = len(self.displacement_actual)
        self.error_side_Var.append(self.error_side_calc(corners))
    
    def displacement_average(self,fname_begin=1,fname_end=0):
        if (fname_end == 0) & (fname_begin == 1):
            return self.displacement_averaged[1:]
        else:
            return self.displacement_averaged[self.idx[fname_begin]+1:self.idx[fname_end]+1]

    def error_side(self,fname_begin=1,fname_end=0):
        if (fname_end == 0) & (fname_begin == 1):
            return self.error_side_Var
        else:
            return self.error_side_Var[self.idx[fname_begin]:self.idx[fname_end]]    
    
    '''
        TODO: create averaged error and ROI methods
    def error_avg():

    def roi():
    '''