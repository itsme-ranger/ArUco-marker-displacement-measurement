import cv2 as cv
import glob
import datetime

URL = '/media/ranger/01D454F10F8A7250/'
imfile = 'DSC02678'
images = glob.glob(URL+'*.JPG')
images.sort()
for i in range(len(images)):
    print(i)
    for j in range(3):
        img = cv.imread(images[i])
        img = cv.edgePreservingFilter(img, sigma_r=0.3, sigma_s=200)
        # index = images[i].index('DSC')
        cv.imwrite(images[i], img)
    # cv.imwrite(URL+URLsample+'edgePreserving1/'+str(i).zfill(4)+'.jpg',img)