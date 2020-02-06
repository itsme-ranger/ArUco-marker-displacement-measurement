# '''/
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
    if len(ids) > 1:
        rvec, tvec = aruco.estimatePoseSingleMarkers([corners[0]], marker_size[ids[1][0]], newcameramtx, dist0)
    else:
        rvec, tvec = aruco.estimatePoseSingleMarkers(corners, marker_size[ids[0][0]], newcameramtx, dist0)
    # rvec, tvec = aruco.estimatePoseSingleMarkers(corners, marker_size,mtx, dist)
    # tvecx = tvec.tolist()
    # for i in tvecx:
    #     for line in i:
    #         # f.write(" ".join(line) + "\n")
    #         f.write(str(line) + " ")

    print("corners:\n" + str(corners) + "\n------\n\n")
    f.write("corners:\n" + str(corners) + "\n")
    # f.write(tvec)
    # It's working.
    # my problem was that the cellphone put black all around it. The alrogithm
    # depends very much upon finding rectangular black blobs

    # gray = aruco.drawDetectedMarkers(gray, rejectedImgPoints)
    # img = aruco.drawDetectedMarkers(img, corners,borderColor=(0, 255, 0))
    # img = cv.resize(img,(int(gray.shape[1]/3),int(gray.shape[0]/3)))
    # corners = [x/3 for x in corners]
    # rejectedImgPoints = [x/3 for x in rejectedImgPoints]
    '''/
    for i in range(0, len(rvec)):
        img = aruco.drawAxis(img, newcameramtx, dist0, rvec[i], tvec[i], marker_size[ids[0][0]])
        # img = aruco.drawAxis(img, mtx, dist, rvec[i], tvec[i], marker_size[ids[0][0]])
    /'''
    # for corner in corners[0][0]:
    corpri = cor_prior
    cor_prior = np.insert(cor_prior, 2, 1, axis=1)
    cor = corners[0][0]
    cor = np.insert(cor, 2, 1, axis=1)

    a = (int(min(corners[0][0][0][1], corners[0][0][3][1])) - crop_margin)
    b = (int(max(corners[0][0][1][1], corners[0][0][2][1])) + crop_margin)
    c = int(min(corners[0][0][0][0], corners[0][0][1][0])) - crop_margin
    d = int(max(corners[0][0][2][0], corners[0][0][3][0])) + crop_margin
    roi = img[a:b, c:d]
    # roi = img[int(min(cor[0][0],cor[3][0]))-crop_margin:int(max(cor[1][0],cor[2][0]))+crop_margin, int(min(cor[0][1],cor[1][1]))-crop_margin:int(max(cor[2][1],cor[3][1]))+crop_margin]
    cv.imwrite(URL + URLsample + 'detectedmarker/' + filename[stridxi:], roi)
    rotmat, jacob = cv.Rodrigues(rvec)
    # wx = rvec[0][0][0]
    # wy = rvec[0][0][1]
    # wz = rvec[0][0][2]
    # tx = tvec[0][0][0]
    # ty = tvec[0][0][1]
    # tz = tvec[0][0][2]
    matr = np.append(rotmat.transpose(), tvec[0], axis=0).transpose()
    # matr = np.matmul(mtx,np.array([[1,-wz,wy,tx],[wz,1,-wx,ty],[-wy,wx,1,tz]]))
    WP = np.zeros([4, 4])
    for i in range(len(cor)):
        mat = np.matmul(inv(newcameramtx), cor_prior[i])
        WP_prior[i] = np.matmul(pinv(matr), mat)
        mat = np.matmul(inv(newcameramtx), cor[i])
        WP[i] = np.matmul(pinv(matr), mat)
    print(WP.shape)
    print(WP)
    WP_center = np.zeros(3)
    ma = (WP[2][1] - WP[0][1]) / (WP[2][0] - WP[0][0])
    mb = (WP[1][1] - WP[3][1]) / (WP[1][0] - WP[3][0])
    WP_center[0] = (WP[3][1] - WP[0][1] + ma * WP[0][0] - mb * WP[3][0]) / (ma - mb)
    WP_center[1] = ma * (WP_center[0] - WP[0][0]) + WP[0][1]
    WP_center[2] = (WP[2][2] - WP[0][2]) / (WP[2][0] - WP[0][0]) * (WP_center[0] - WP[0][0]) + WP[0][2]
    # WP_center[0] = (WP[0][1])/((WP[1][1]-WP[3][1])/(WP[1][0]-WP[3][0])-(WP[2][1]-WP[0][1])/(WP[2][0]-WP[0][0]))
    # WP_center[1] = (WP[1][1]-WP[3][1])/(WP[1][0]-WP[3][0])*WP_center[0]
    # WP_center[2] = (WP[1][2]-WP[3][2])/(WP[1][0]-WP[3][0])*WP_center[0]
    print(WP_center)

    ma = (WP_prior[2][1] - WP_prior[0][1]) / (WP_prior[2][0] - WP_prior[0][0])
    mb = (WP_prior[1][1] - WP_prior[3][1]) / (WP_prior[1][0] - WP_prior[3][0])
    WP_center_prior[0] = (WP_prior[3][1] - WP_prior[0][1] + ma * WP_prior[0][0] - mb * WP_prior[3][0]) / (ma - mb)
    WP_center_prior[1] = ma * (WP_center_prior[0] - WP_prior[0][0]) + WP_prior[0][1]
    WP_center_prior[2] = (WP_prior[2][2] - WP_prior[0][2]) / (WP_prior[2][0] - WP_prior[0][0]) * (
                WP_center_prior[0] - WP_prior[0][0]) + WP_prior[0][2]

    # WP_center_prior[0] = (WP_prior[0][1]) / ((WP_prior[1][1] - WP_prior[3][1]) / (WP_prior[1][0] - WP_prior[3][0]) - (WP_prior[2][1] - WP_prior[0][1]) / (WP_prior[2][0] - WP_prior[0][0]))
    # WP_center_prior[1] = (WP_prior[1][1] - WP_prior[3][1]) / (WP_prior[1][0] - WP_prior[3][0]) * WP_center_prior[0]
    # WP_center_prior[2] = (WP_prior[1][2] - WP_prior[3][2]) / (WP_prior[1][0] - WP_prior[3][0]) * WP_center_prior[0]
    f.write("\njarak:\n")
    # print(WP)
    jarak_rata = 0
    jarak_max = 0
    jarak_min = 1000
    for i in range(len(WP)):
        distX = WP[i][0] - WP_prior[i][0]
        distY = WP[i][1] - WP_prior[i][1]
        distZ = WP[i][2] - WP_prior[i][2]
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
        jarak_max - jarak_min) + "px\n")  # " kecermatan = " + str(marker_size[ids[0][0]] / jarak_rata)
    db.write(str(jarak_min) + ',' + str(jarak_max) + ',' + str(jarak_rata) + ',')
    distX = WP_center[0] - WP_center_prior[0]
    distY = WP_center[1] - WP_center_prior[1]
    distZ = WP_center[2] - WP_center_prior[2]
    distYZctr = math.sqrt(distY ** 2 + distZ ** 2)
    f.write("center\nX = " + str(distX) + "; Z = " + str(distZ) + "; Y = " + str(distY) + "\ndistYZ center = " + str(
        distYZctr) + "\n")
    jarak_rata = 0
    jarak_max = 0
    jarak_min = 10000
    for i in range(4):
        new = distance(WP[i], WP[(i + 1) % 4], [1, 1, 1, 0])
        jarak_min = min(jarak_min, new)
        jarak_max = max(jarak_max, new)
        jarak = new
        jarak_rata += new / 4
        f.write("\ndist corner WP" + str(i) + "-" + str((i + 1) % 4) + "= " + str(jarak))
    jangkauan = jarak_max - jarak_min
    jarak_total = distance(WP_center_prior, WP_center, [1, 1, 1, 0]) * marker_size[ids[0][0]] / jarak_rata
    kecermatan = marker_size[ids[0][0]] / jarak_rata
    f.write("\njarak corner rata2 WP = " + str(jarak_rata) + " kecermatan = " + str(
        kecermatan) + "mm/px jangkauan = " + str(jangkauan) + "px\nSetelah dinormalisasi, X= " + str(
        distance(WP_center_prior, WP_center, [1, 0, 0]) * marker_size[ids[0][0]] / jarak_rata) + "; Y = " + str(
        distY * marker_size[ids[0][0]] / jarak_rata) + "; distYZ center = " + str(
        distYZctr * marker_size[ids[0][0]] / jarak_rata) + "; Z=" + str(
        distance(WP_center_prior, WP_center, [0, 0, 1, 0]) * marker_size[ids[0][0]] / jarak_rata) + "jarak total= " + str(
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
        marker_size[ids[0][0]] / jarak_rata) + "mm/px jangkauan = " + str(
        jarak_max - jarak_min) + "px\n")

    # WP_prior = WP
    # WP_center_prior = WP_center
    WPp_RTp = cornerstoWP(corpri, newcameramtx, rvecp, tvecp[0])
    WP_center_prior = WPofCenter(WPp_RTp)
    WP_RTp = cornerstoWP(corners[0][0], newcameramtx, rvecp, tvecp[0])
    WP_center = WPofCenter(WP_RTp)
    distX = abs(WP_center[0] - WP_center_prior[0])
    distY = abs(WP_center[1] - WP_center_prior[1])
    distZ = abs(WP_center[2] - WP_center_prior[2])
    distYZctr = math.sqrt(distY ** 2 + distZ ** 2)
    f.write("RT old: center\nX = " + str(distX) + "; Z = " + str(distZ) + "; Y = " + str(
        distY) + "\ndistYZ center = " + str(
        distYZctr) + "\nSetelah dinormalisasi, Y = " + str(
        distY * marker_size[ids[0][0]] / jarak_rata) + "; distYZ center = " + str(distYZctr * marker_size[ids[0][0]] / jarak_rata) + "\n")
    # /'''
