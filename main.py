import cv2 as cv
import cv2.aruco as aruco
import glob
from .config import *
from .markercalc import Marker

def load_img(folder,extension):
    images = glob.glob(folder+'*.'+extension)
    images.sort()
    return images

def initMarkers(marker_size):
    markers = {}
    for _id,size in marker_size.items():
        markers[_id] = Marker(_id,size)
    return markers

def detectMarkers(gray, filename):
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    for i,_id in enumerate(ids):
        if (_id[0] not in markers) & (_id[0] in marker_size):
            markers[_id[0]] = Marker(_id[0])
        markers[_id[0]](corners[i][0],filename=filename)
    return markers


# MEASUREMENT PROCESS
images = load_img(folder,extension)
markers = initMarkers(marker_size)

for image in images:
    gray = cv.imread(image,0)
    markers = detectMarkers(gray,image)

# RESULTS ANALYSIS
for marker in markers.values():
    print(marker.displacement_average())