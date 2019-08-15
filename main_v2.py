import cv2 as cv

def cornerstoWP(corners,rvec,tvec):
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

def calib(url,extension,chessbrd_size,criteria,calib_resize):
    images = glob.glob(url + '*.' + extension)
    for filename in images:
        img = cv.imread(filename)
        img = cv.resize(img,(int(img.shape[1]/calib_resize),int(img.shape[0]/calib_resize)))
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        print("berhasil input 1,5 nih " + str(i))
        ret, corners = cv.findChessboardCorners(gray, chessbrd_size,
                                                flags=cv.CALIB_CB_ADAPTIVE_THRESH)  # | cv.CALIB_CB_NORMALIZE_IMAGE | cv.CALIB_CB_FILTER_QUADS | cv.CALIB_CB_FAST_CHECK

        if ret == True:
            objpoints.append(objp)

            cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)
            print("berhasil input 3 " + str(i))

            # nunjukin cornernya
            # cv.drawChessboardCorners(img,chessbrd_size, corners,ret)
            # cv.imshow('img',img)
            # cv.waitKey()
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    return newcameramtx,mtx,dist

def arucoCustomParam(x):
    parameters = aruco.DetectorParameters_create()
    if x == 1:
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
        setattr(parameters, 'adaptiveThreshWinSizeMax', int(minWin + WinSize * maxWin))
        setattr(parameters, 'adaptiveThreshWinSizeStep', stepsizeHL)
        # setattr(parameters,'polygonalApproxAccuracyRate',0.21)
    return parameters

def measurement(url,textfiles,extension,initMeasurement,intrinsic,parameters):
    f,db = textfiles
    images = glob.glob(url+ '*.'+extension)
    images.sort()
    newcameramtx, mtx, dist = intrinsic
    locals().update(initMeasurement)

    for filename in images:
        stridxi = filename.index('DSC')
        stridxo = filename.index('.'+str(extension))
        db.write(filename[stridxi:stridxo] + ',')
        print(filename)
        img = cv.imread(filename)
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        img = cv.undistort(img, mtx, dist, None, newcameramtx)
        f.write("\n\n" + filename + "\n")
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # lists of ids and the corners beloning to each id
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        print(corners)
        if len(ids) > 1:
            img = aruco.drawDetectedMarkers(img, [corners[0]], borderColor=(0, 255, 0))
        else:
            img = aruco.drawDetectedMarkers(img, corners, borderColor=(0, 255, 0))

        # check if the detected marker is more than 1
        # '''/
        if len(ids) > 1:
            rvec, tvec = aruco.estimatePoseSingleMarkers([corners[0]], marker_size, newcameramtx, dist0)
        else:
            rvec, tvec = aruco.estimatePoseSingleMarkers(corners, marker_size, newcameramtx, dist0)

        print("corners:\n" + str(corners) + "\n------\n\n")
        f.write("corners:\n" + str(corners) + "\n")

        # '''/
        for i in range(0, len(rvec)):
            img = aruco.drawAxis(img, newcameramtx, dist0, rvec[i], tvec[i], marker_size)
            # img = aruco.drawAxis(img, mtx, dist, rvec[i], tvec[i], marker_size)
        # /'''
        # for corner in corners[0][0]:
        corpri = cor_prior
        cor_prior = np.insert(cor_prior, 2, 1, axis=1)
        cor = corners[0][0]
        cor = np.insert(cor, 2, 1, axis=1)

        # create image file from ROI of marker
        a = (int(min(corners[0][0][0][1], corners[0][0][3][1])) - crop_margin)
        b = (int(max(corners[0][0][1][1], corners[0][0][2][1])) + crop_margin)
        c = int(min(corners[0][0][0][0], corners[0][0][1][0])) - crop_margin
        d = int(max(corners[0][0][2][0], corners[0][0][3][0])) + crop_margin
        roi = img[a:b, c:d]

        cv.imwrite(url+ 'detectedmarker/' + filename[stridxi:], roi)
        rotmat, jacob = cv.Rodrigues(rvec)

        matr = np.append(rotmat.transpose(), tvec[0], axis=0).transpose()

        WP = np.zeros([4, 4])
        for i in range(len(cor)):
            WP_prior[i] = np.matmul(pinv(matr), cor_prior[i])
            WP[i] = np.matmul(pinv(matr), cor[i])
        print(WP.shape)
        print(WP)
        WP_center = np.zeros(3)
        ma = (WP[2][1] - WP[0][1]) / (WP[2][0] - WP[0][0])
        mb = (WP[1][1] - WP[3][1]) / (WP[1][0] - WP[3][0])
        WP_center[0] = (WP[3][1] - WP[0][1] + ma * WP[0][0] - mb * WP[3][0]) / (ma - mb)
        WP_center[1] = ma * (WP_center[0] - WP[0][0]) + WP[0][1]
        WP_center[2] = (WP[2][2] - WP[0][2]) / (WP[2][0] - WP[0][0]) * (WP_center[0] - WP[0][0]) + WP[0][2]
        print(WP_center)

        ma = (WP_prior[2][1] - WP_prior[0][1]) / (WP_prior[2][0] - WP_prior[0][0])
        mb = (WP_prior[1][1] - WP_prior[3][1]) / (WP_prior[1][0] - WP_prior[3][0])
        WP_center_prior[0] = (WP_prior[3][1] - WP_prior[0][1] + ma * WP_prior[0][0] - mb * WP_prior[3][0]) / (ma - mb)
        WP_center_prior[1] = ma * (WP_center_prior[0] - WP_prior[0][0]) + WP_prior[0][1]
        WP_center_prior[2] = (WP_prior[2][2] - WP_prior[0][2]) / (WP_prior[2][0] - WP_prior[0][0]) * (
                    WP_center_prior[0] - WP_prior[0][0]) + WP_prior[0][2]

        f.write("\njarak:\n")
        jarak_rata = 0
        jarak_max = 0
        jarak_min = 1000
        for i in range(len(WP)):
            distX = abs(WP[i][0] - WP_prior[i][0])
            distY = abs(WP[i][1] - WP_prior[i][1])
            distZ = abs(WP[i][2] - WP_prior[i][2])
            jarake = math.sqrt(sum([(a - b) ** 2 for a, b in zip(WP[i], WP_prior[i])]))
            distYZ = math.sqrt(distY ** 2 + distZ ** 2)
            f.write("X = " + str(distX) + "; Z = " + str(distZ) + "; Y = " + str(distY) + "\ndist = " + str(
                jarake) + "\n" + "distYZ = " + str(distYZ) + "\n")
            # mengukur jarak selisih antara corner dan corner sebelumnya
            new = distance(cor[i], cor_prior[i], [1, 1, 0])
            jarak_min = min(jarak_min, new)
            jarak_max = max(jarak_max, new)
            jarak = new
            jarak_rata += new / 4
            f.write("dist corner dengan corner sebelumnya\ndist corner px" + str(i) + "= " + str(jarak))
        f.write("\njarak corner rata2 px = " + str(jarak_rata) + "mm/px jangkauan = " + str(
            jarak_max - jarak_min) + "px\n")  # " kecermatan = " + str(marker_size / jarak_rata)
        db.write(str(jarak_min) + ',' + str(jarak_max) + ',' + str(jarak_rata) + ',')
        distX = abs(WP_center[0] - WP_center_prior[0])
        distY = abs(WP_center[1] - WP_center_prior[1])
        distZ = abs(WP_center[2] - WP_center_prior[2])
        distYZctr = math.sqrt(distY ** 2 + distZ ** 2)
        f.write(
            "center\nX = " + str(distX) + "; Z = " + str(distZ) + "; Y = " + str(distY) + "\ndistYZ center = " + str(
                distYZctr) + "\n")
        jarak_rata = 0
        jarak_max = 0
        jarak_min = 1000
        for i in range(4):
            new = distance(WP[i], WP[(i + 1) % 4], [1, 1, 1, 0])
            jarak_min = min(jarak_min, new)
            jarak_max = max(jarak_max, new)
            jarak = new
            jarak_rata += new / 4
            f.write("\ndist corner WP" + str(i) + "-" + str((i + 1) % 4) + "= " + str(jarak))
        jangkauan = jarak_max - jarak_min
        jarak_total = distance(WP_center_prior, WP_center, [1, 1, 1, 0]) * marker_size / jarak_rata
        kecermatan = marker_size / jarak_rata
        f.write("\njarak corner rata2 WP = " + str(jarak_rata) + " kecermatan = " + str(
            kecermatan) + "mm/px jangkauan = " + str(jangkauan) + "px\nSetelah dinormalisasi, X= " + str(
            distance(WP_center_prior, WP_center, [1, 0, 0]) * marker_size / jarak_rata) + "; Y = " + str(
            distY * marker_size / jarak_rata) + "; distYZ center = " + str(
            distYZctr * marker_size / jarak_rata) + "; Z=" + str(
            distance(WP_center_prior, WP_center, [0, 0, 1, 0]) * marker_size / jarak_rata) + "jarak total= " + str(
            jarak_total) + "\n")
        db.write(str(jarak_min) + ',' + str(jarak_max) + ',' + str(jarak_rata) + ',' + str(jangkauan) + ',' + str(
            kecermatan) + ',' + str(jarak_total))
        # mengukur jarak antar-corner dalam unit image plane
        jarak_rata = 0
        jarak_max = 0
        jarak_min = 1000
        for i in range(4):
            new = distance(corners[0][0][i], corners[0][0][(i + 1) % 4], [1, 1])
            jarak_min = min(jarak_min, new)
            jarak_max = max(jarak_max, new)
            jarak = new
            jarak_rata += new / 4
            f.write("\ndist corner px" + str(i) + "-" + str((i + 1) % 4) + "= " + str(jarak))
        f.write("\njarak corner rata2 px = " + str(jarak_rata) + " kecermatan = " + str(
            marker_size / jarak_rata) + "mm/px jangkauan = " + str(
            jarak_max - jarak_min) + "px\n")

        WPp_RTp = cornerstoWP(corpri, rvecp, tvecp)
        WP_center_prior = WPofCenter(WPp_RTp)
        WP_RTp = cornerstoWP(corners[0][0], rvecp, tvecp)
        WP_center = WPofCenter(WP_RTp)
        distX = abs(WP_center[0] - WP_center_prior[0])
        distY = abs(WP_center[1] - WP_center_prior[1])
        distZ = abs(WP_center[2] - WP_center_prior[2])
        distYZctr = math.sqrt(distY ** 2 + distZ ** 2)
        f.write("RT old: center\nX = " + str(distX) + "; Z = " + str(distZ) + "; Y = " + str(
            distY) + "\ndistYZ center = " + str(
            distYZctr) + "\nSetelah dinormalisasi, Y = " + str(
            distY * marker_size / jarak_rata) + "; distYZ center = " + str(distYZctr * marker_size / jarak_rata) + "\n")
        f.write("\n-------\n\n")

        # STAGE 2
        # MEASUREMENT BASED ON DISTANCE RELATIVE FROM MARKER REFERENCE
        cornersdict = {}
        for x in range(len(corners)):
            if ids[x] in markers_id:
                cornersdict[ids[x]] = corners[x]

        WP_RTp = cornerstoWP(corners[0][0], rvecp, tvecp)

        rvecp = rvec
        tvecp = tvec
        cor_prior = corners[0][0]
        db.write('\n')

# Initialize Measurement Parameters
initMeasurement = {
    # parameter you need to input
    "aruco_dict" : aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL),
    "marker_size" : 200,  # length of marker side in mm
    "crop_margin" : 50  # how much pixel as margin from the detected marker
    "markers_id" : [100,110,120,130,140] # marker's ID which to be used for the measurement, the first one will be used as marker's reference
    "cor_prior" : np.random.rand(4, 2),  # create dummy corners for first calculation
    "WP_prior" : np.zeros([4, 4]),
    "WP_center_prior" : np.zeros(3),
    "matr_prior" : np.append(np.identity(3), [[0, 0, 0]], axis=0).transpose(),
    "dist0" : np.zeros([1, 5]),
    "rvecp" : np.array([[[1, 1, 1]]], dtype=np.float32),
    "tvecp" : np.array([[[0, 0, 0]]], dtype=np.float32)
}

chessbrd_wide = 20 # 26, 20.5
calib_resize = 1
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, chessbrd_wide, 0.001)
chessbrd_size = (9,6)

URL = '/media/ranger/01D454F10F8A7250/OneDrive - Institut Teknologi Bandung/CAMPUSS_S1/TAnjink/'
URLsample = 'lab/try13/edgePreserving4/'
images = URL+URLsample
extension = 'JPG'
URLstrip = ['-' if x == '/' else x for x in URLsample]
URLstrip = ''.join(URLstrip)
now = datetime.datetime.now()
f = open("invoice "+URLstrip+' '+str(now.strftime("%m-%d_%H-%M"))+".txt","a+")
db = open("db "+URLstrip+' '+str(now.strftime("%m-%d_%H-%M"))+".csv","a+")
db.write('\n\n'+str(URLsample)+'\n'+str(datetime.datetime.now())+'\n')
f.write("\n\n----------xxxxxxx------xxxxxx-------\n\n\n")
f.write(str(datetime.datetime.now()))
f.write(str(now.strftime("%m-%d_%H-%M")))
textfiles = [f,db]

# main
newcameramtx,mtx,dist = calib(images,extension,chessbrd_size,criteria,calib_resize)
intrinsic = [newcameramtx,mtx,dist]
parameters = arucoCustomParam(1)
measurement(images,textfiles,extension,initMeasurement,intrinsic,parameters)
f.close()
db.close()
print(c)
print(datetime.datetime.now())
print('selesai')