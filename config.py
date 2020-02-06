# IMAGE FOLDER AND EXTENSION
folder = 'D:/aku/sayang/kamu' # use full folder URL with tree separation '/' instead '\'
extension = '.jpg' # case-sensitive

# ARUCO MARKER PARAMETERS
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
setattr(parameters, 'cornerRefinementWinSize', WinSize)
setattr(parameters, 'cornerRefinementMinAccuracy', minAcc)
setattr(parameters, 'cornerRefinementMaxIterations', maxIter)
setattr(parameters, 'adaptiveThreshWinSizeMin', minWin)
setattr(parameters, 'adaptiveThreshWinSizeMax', int(minWin + WinSize * maxWin))
setattr(parameters, 'adaptiveThreshWinSizeStep', stepsizeHL)

marker_size = { # length of marker side in mm with marker's id as keys
    14:140,
    13:105}

# INTRINSIC INITIALIZATION
dist = [[9.53306186e-03, -5.61930848e-02, -8.66880870e-03, -1.60162599e-03, -5.23222839e+00]]
mtx = [[1.30822305e+04, 0.00000000e+00, 2.95153154e+03],
       [0.00000000e+00, 1.30659778e+04, 1.68524545e+03],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
dist = np.asarray(dist)
mtx = np.asarray(mtx)