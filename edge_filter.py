import cv2 as cv
import glob
import datetime

URL = '/media/ranger/01D454F10F8A7250/OneDrive - Institut Teknologi Bandung/CAMPUSS_S1/TAnjink/'
URLsample = 'lab/try13/'
imfile = 'DSC02678'
print(datetime.datetime.now())
c = datetime.datetime.now()
'''/
images = glob.glob(URL+'kalibrasi/A6000/50mm-2/edgePreserving1/*.jpg')
for i in range(len(images)):
    print(i)
    img = cv.imread(images[i])
    img = cv.edgePreservingFilter(img, sigma_r=1)
    cv.imwrite(URL + 'kalibrasi/A6000/50mm-2/edgePreserving2/' + str(i).zfill(4) + '.jpg', img)
    # cv.imwrite(URL+URLsample+'edgePreserving/'+str(i)+'.jpg',img)
/'''
'''/
images = glob.glob(URL+URLsample+'edgePreserving4/*.JPG')
for i in range(len(images)):
    print(i)
    img = cv.imread(images[i])
    img = cv.edgePreservingFilter(img, sigma_r=1)
    index = images[i].index('DSC')
    cv.imwrite(URL + URLsample + 'edgePreserving2/' + images[i][index:], img)
    # cv.imwrite(URL+URLsample+'edgePreserving1/'+str(i).zfill(4)+'.jpg',img)
/'''
'''/
for j in range(1,6):
    print(j)
    print(datetime.datetime.now())
    images = glob.glob(URL+URLsample+'edgePreserving'+str(j)+'/*.JPG')
    for i in range(len(images)):
        img = cv.imread(images[i])
        img = cv.edgePreservingFilter(img, sigma_r=0.1,sigma_s=100)
        img = cv.edgePreservingFilter(img, sigma_r=1)
        index = images[i].index('DSC')
        cv.imwrite(URL + URLsample + 'edgePreserving'+str(j+1)+'/' + images[i][index:], img)
    # cv.imwrite(URL+URLsample+'edgePreserving1/'+str(i).zfill(4)+'.jpg',img)
/'''
# folder = 'DSC02678 ef sigma r max/'
'''/
folder = 'DSC02678 ef ss200 sr03/'
img = cv.imread(URL+URLsample+imfile+'.JPG')
img = cv.edgePreservingFilter(img, sigma_r=0.3, sigma_s=200)
cv.imwrite(URL + URLsample+folder+imfile+'-ss200 sr0.3-0.JPG',img)
for j in range(6):
    print(j)
    print(datetime.datetime.now())
    img = cv.imread(URL+URLsample+folder+imfile+'-ss200 sr0.3-'+str(j)+'.JPG')
    img = cv.edgePreservingFilter(img, sigma_r=0.3, sigma_s=200)
    cv.imwrite(URL + URLsample+folder+imfile+'-ss200 sr0.3-'+str(j+1)+'.JPG',img)
d= (datetime.datetime.now())
folder = 'DSC02678 ef ss200 sr04/'
img = cv.imread(URL+URLsample+imfile+'.JPG')
img = cv.edgePreservingFilter(img, sigma_r=0.4, sigma_s=200)
cv.imwrite(URL + URLsample+folder+imfile+'-ss200 sr0.4-0.JPG',img)
for j in range(6):
    print(j)
    print(datetime.datetime.now())
    img = cv.imread(URL+URLsample+folder+imfile+'-ss200 sr0.4-'+str(j)+'.JPG')
    img = cv.edgePreservingFilter(img, sigma_r=0.4, sigma_s=200)
    cv.imwrite(URL + URLsample+folder+imfile+'-ss200 sr0.4-'+str(j+1)+'.JPG',img)
folder = 'DSC02678 ef ss200 sr05/'
img = cv.imread(URL+URLsample+imfile+'.JPG')
img = cv.edgePreservingFilter(img, sigma_r=0.5, sigma_s=200)
cv.imwrite(URL + URLsample+folder+imfile+'-ss200 sr0.5-0.JPG',img)
for j in range(6):
    print(j)
    print(datetime.datetime.now())
    img = cv.imread(URL+URLsample+folder+imfile+'-ss200 sr0.5-'+str(j)+'.JPG')
    img = cv.edgePreservingFilter(img, sigma_r=0.5, sigma_s=200)
    cv.imwrite(URL + URLsample+folder+imfile+'-ss200 sr0.5-'+str(j+1)+'.JPG',img)
folder = 'DSC02678 ef ss200 sr06/'
img = cv.imread(URL+URLsample+imfile+'.JPG')
img = cv.edgePreservingFilter(img, sigma_r=0.6, sigma_s=200)
cv.imwrite(URL + URLsample+folder+imfile+'-ss200 sr0.6-0.JPG',img)
for j in range(6):
    print(j)
    print(datetime.datetime.now())
    img = cv.imread(URL+URLsample+folder+imfile+'-ss200 sr0.6-'+str(j)+'.JPG')
    img = cv.edgePreservingFilter(img, sigma_r=0.6, sigma_s=200)
    cv.imwrite(URL + URLsample+folder+imfile+'-ss200 sr0.6-'+str(j+1)+'.JPG',img)

'''
'''/
folder = 'DSC02678 ef sigma s max/'
for j in range(6):
    print(j)
    print(datetime.datetime.now())
    img = cv.imread(URL+URLsample+folder+imfile+'-'+str(j)+'.JPG')
    img = cv.edgePreservingFilter(img, sigma_r=0.05, sigma_s=200)
    cv.imwrite(URL + URLsample+folder+imfile+'-'+str(j+1)+'.JPG',img)
/'''
images = glob.glob(URL+URLsample+'edgePreserving4/*.JPG')
images.sort()
for i in range(len(images)):
    print(i)
    for j in range(3):
        img = cv.imread(images[i])
        img = cv.edgePreservingFilter(img, sigma_r=0.3, sigma_s=200)
        # index = images[i].index('DSC')
        cv.imwrite(images[i], img)
    # cv.imwrite(URL+URLsample+'edgePreserving1/'+str(i).zfill(4)+'.jpg',img)
print('\n')
print(c)
print d
print(datetime.datetime.now())
print ('selesai')