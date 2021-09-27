import cv2
import mediapipe as mp
import time
import pandas as pd
import numpy as np
import gb_utils
from db import add_movement_record
from model import  *


def sit2stand(video, ppm):
    print("In sit2stand")
    mpDraw = mp.solutions.drawing_utils
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()

    cap = cv2.VideoCapture(video)
    pTime = 0

    # setting video output
    # out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, (int(cap.get(3)), int  (cap.get(4))))

    pos_all = []

    parts = [0, 11, 12, 24, 23, 5, 2, 8, 7]

    start = time.time()
    while cap.isOpened():
        success, img = cap.read()
        if success:
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(imgRGB)
            # results = pose.process(img)
            #print(results.pose_landmarks)
            points = []
            if results.pose_landmarks:
                mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    h, w, c = img.shape
                    print(id, lm)
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    points.append([cx, cy])
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

            # cTime = time.time()
            # fps = 1 / (cTime - pTime)
            # pTime = cTime

            # cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

            # processing sit2stand
            rols = []
            for part in parts:
                try:
                    rols.append(points[part][1])
                except:
                    rols.append(float('nan'))
            pos_all.append(np.nanmean(rols))
            print(rols)

            # Write the frame into the file 'output.mp4'

            # out.write(img)

            # cv2.imshow("Image", img)
            cv2.waitKey(1)
        else:
            break

    end = time.time()
    print("Analysis time: " + str(end - start) + " seconds")

    # Filter the results
    pos_med = gb_utils.filterG(pos_all)
    filt = gb_utils.lowpass(pos_med, 2)

    reps = gb_utils.rep_count(filt)

    # STORE RESULTS TO DF
    data = pd.DataFrame({'pos_all': pos_all, 'pos_med': pos_med, 'filt': filt, 'Reps': reps})
    s2s = Sit2stand(int(data["Reps"].max()))
    print("Rs in s2s: ", s2s)
    # status = add_movement_record(move, s2s, email)
    
    return s2s


def hip_abduction(video, ppm):
    mpDraw = mp.solutions.drawing_utils
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()

    cap = cv2.VideoCapture(video)
    pTime = 0

    # setting video output
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, (int(cap.get(3)), int  (cap.get(4))))

    lr_dist_all = []
    hip_r_all = []
    hip_l_all = []

    start = time.time()
    while cap.isOpened():
        success, img = cap.read()
        if success:
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(imgRGB)
            # results = pose.process(img)
            #print(results.pose_landmarks)
            points = []
            if results.pose_landmarks:
                mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    h, w, c = img.shape
                    print(id, lm)
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    points.append([cx, cy])
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

            # processing hip abduction
            lr_dist_all.append(gb_utils.distance(points[28][0], points[28][1],
                                                 points[27][0], points[27][1]))
            hip_r_all.append(360 - (gb_utils.getAngle(points[26], points[24], points[23])))
            hip_l_all.append(gb_utils.getAngle(points[25], points[23], points[24]))

            # Write the frame into the file 'output.mp4'
            # cv2.imwrite('frame.jpg', img)
            # out.write(img)
            # cv2.imshow("Image", img)
            cv2.waitKey(1)
        else:
            break

    end = time.time()
    print("Analysis time: " + str(end - start) + " seconds")

     # Filter the results
    lr_med = gb_utils.filterG(lr_dist_all)
    hip_r_med = gb_utils.filterG(hip_r_all)
    hip_l_med = gb_utils.filterG(hip_l_all)

    # STORE RESULTS TO DF

    data = pd.DataFrame({'lr_med': lr_med, 'hip_r_med': hip_r_med,'hip_l_med': hip_l_med})

    data['lr_med'] = data['lr_med']/ppm

    dmax = data.max()
    res = HipAbduction(dmax['lr_med'], dmax['hip_r_med'], dmax['hip_l_med'])

    return res


def cerv_rot(video, ppm):
    mpDraw = mp.solutions.drawing_utils
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()

    cap = cv2.VideoCapture(video)
    pTime = 0

    # setting video output
    # out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, (int(cap.get(3)), int  (cap.get(4))))

    l_dist_all = []
    r_dist_all = []

    start = time.time()
    while cap.isOpened():
        success, img = cap.read()
        if success:
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(imgRGB)
            # results = pose.process(img)
            #print(results.pose_landmarks)
            points = []
            if results.pose_landmarks:
                mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    h, w, c = img.shape
                    print(id, lm)
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    points.append([cx, cy])
                    # cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            # cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

            # processing cerv_rot
            r_dist_all.append(np.abs(points[0][0] - points[8][0]))
            l_dist_all.append(np.abs(points[0][0] - points[7][0]))

            # Write the frame into the file 'output.mp4'
            # cv2.imwrite('frame.jpg', img)
            # out.write(img)
            # cv2.imshow("Image", img)
            cv2.waitKey(1)
        else:
            break

    end = time.time()
    print("Analysis time: " + str(end - start) + " seconds")

    # Filter the results
    l_med = gb_utils.filterG(l_dist_all)
    r_med = gb_utils.filterG(r_dist_all)

    # STORE RESULTS TO DF

    data = pd.DataFrame({'l_med': l_med, 'r_med': r_med})
    data['l_med']= data['l_med']/ppm
    data['r_med']= data['r_med']/ppm

    dmax = data.max()
    cerv = CervRot(dmax['l_med'], dmax['r_med'])

    return cerv


def hip_int_rot(video, ppm):
    mpDraw = mp.solutions.drawing_utils
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()

    cap = cv2.VideoCapture(video)
    pTime = 0

    # setting video output
    # out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, (int(cap.get(3)), int  (cap.get(4))))

    l_mid_all = []
    r_mid_all = []
    l_r_all = []

    start = time.time()
    while cap.isOpened():
        success, img = cap.read()
        if success:
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(imgRGB)
            # results = pose.process(img)
            #print(results.pose_landmarks)
            points = []
            if results.pose_landmarks:
                mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    h, w, c = img.shape
                    print(id, lm)
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    points.append([cx, cy])
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

            # processing hip int rot
            mid = (points[13][0] - points[26][0]) / 2 + points[26][0]
            r_mid_all.append(np.abs(points[28][0] - mid))
            l_mid_all.append(np.abs(points[27][0] - mid))
            l_r_all.append(np.abs(points[27][0] - points[28][0]))

            # Write the frame into the file 'output.mp4'
            # cv2.imwrite('frame.jpg', img)
            # out.write(img)
            # cv2.imshow("Image", img)
            cv2.waitKey(1)
        else:
            break

    end = time.time()
    print("Analysis time: " + str(end - start) + " seconds")

    # Filter the results
    l_med = gb_utils.filterG(l_mid_all)
    r_med = gb_utils.filterG(r_mid_all)
    lr_med = gb_utils.filterG(l_r_all)

    # STORE RESULTS TO DF

    data = pd.DataFrame({'l_med': l_med,'r_med': r_med,'lr_med': lr_med})


    data['l_med']= data['l_med']/ppm
    data['r_med']= data['r_med']/ppm
    data['lr_med']= data['lr_med']/ppm

    res = data.max()

    hir = HipRot(res['lr_med'], res['l_med'], res['r_med'])

    return hir


def lumbar_flex(video, ppm):
    mpDraw = mp.solutions.drawing_utils
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()

    cap = cv2.VideoCapture(video)
    pTime = 0

    # setting video output
    # out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, (int(cap.get(3)), int  (cap.get(4))))

    l_dist_all = []
    r_dist_all = []

    start = time.time()
    while cap.isOpened():
        success, img = cap.read()
        if success:
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(imgRGB)
            # results = pose.process(img)
            #print(results.pose_landmarks)
            points = []
            if results.pose_landmarks:
                mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    h, w, c = img.shape
                    print(id, lm)
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    points.append([cx, cy])
                    # cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            # cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

            # processing lumbar flex
            r_dist_all.append(points[16][1] - points[28][1])
            l_dist_all.append(points[15][1] - points[27][1])

            # Write the frame into the file 'output.mp4'
            # cv2.imwrite('frame.jpg', img)
            # out.write(img)
            # cv2.imshow("Image", img)
            cv2.waitKey(1)
        else:
            break

    end = time.time()
    print("Analysis time: " + str(end - start) + " seconds")

    # Filter the results
    l_med = gb_utils.filterG(l_dist_all)
    r_med = gb_utils.filterG(r_dist_all)

    # STORE RESULTS TO DF
    data = pd.DataFrame({'l_med': l_med, 'r_med': r_med})
    data['l_med']= data['l_med']/ppm
    data['r_med']= data['r_med']/ppm

    res = data.min()

    print("RES: ", res)

    lflex = LumbarFlex(res['l_med'], res['r_med'])

    return lflex


def shoulder_flex(video, ppm):
    mpDraw = mp.solutions.drawing_utils
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()

    cap = cv2.VideoCapture(video)
    pTime = 0

    # setting video output
    # out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, (int(cap.get(3)), int  (cap.get(4))))

    l_dist_all = []
    r_dist_all = []

    start = time.time()
    while cap.isOpened():
        success, img = cap.read()
        if success:
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(imgRGB)
            # results = pose.process(img)
            #print(results.pose_landmarks)
            points = []
            if results.pose_landmarks:
                mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    h, w, c = img.shape
                    print(id, lm)
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    points.append([cx, cy])
                    # cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            # cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

            # processing shoulder flex
            r_dist_all.append(gb_utils.lineAngle(points[14][0], points[14][1],
                                                 points[12][0], points[12][1]))
            l_dist_all.append(gb_utils.lineAngle(points[13][0], points[13][1],
                                                 points[11][0], points[11][1]))


            # Write the frame into the file 'output.mp4'
            # cv2.imwrite('frame.jpg', img)
            # out.write(img)
            # cv2.imshow("Image", img)
            cv2.waitKey(1)
        else:
            break

    end = time.time()
    print("Analysis time: " + str(end - start) + " seconds")

    # Filter the results
    l_med = gb_utils.filterG(l_dist_all)
    r_med = gb_utils.filterG(r_dist_all)

    # STORE RESULTS TO DF
    data = pd.DataFrame({'l_med': l_med, 'r_med': r_med})

    res = data.max()
    print("Res: ", res)
    sflex = ShoulderFlex(res['l_med'], res['r_med'])

    return sflex


def side_flex(video, ppm):
    mpDraw = mp.solutions.drawing_utils
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()

    cap = cv2.VideoCapture(video)
    pTime = 0

    # setting video output
    # out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, (int(cap.get(3)), int  (cap.get(4))))

    l_dist_all = []
    r_dist_all = []

    start = time.time()
    while cap.isOpened():
        success, img = cap.read()
        if success:
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(imgRGB)
            # results = pose.process(img)
            #print(results.pose_landmarks)
            points = []
            if results.pose_landmarks:
                mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    h, w, c = img.shape
                    print(id, lm)
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    points.append([cx, cy])
                    # cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            # cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

            # processing side flex
            r_dist_all.append(points[16][1] - points[26][1])
            l_dist_all.append(points[15][1] - points[25][1])


            # Write the frame into the file 'output.mp4'
            # cv2.imwrite('frame.jpg', img)
            # out.write(img)
            # cv2.imshow("Image", img)
            cv2.waitKey(1)
        else:
            break

    end = time.time()
    print("Analysis time: " + str(end - start) + " seconds")

    # Filter the results
    l_med = gb_utils.filterG(l_dist_all)
    r_med = gb_utils.filterG(r_dist_all)

    # STORE RESULTS TO DF
    data = pd.DataFrame({'l_med': l_med, 'r_med': r_med})
    data['l_med'] = data['l_med']/ppm
    data['r_med'] = data['r_med']/ppm

    res = data.min()

    print("Res: ", res)
    sdflex = SideFlex(res['l_med'], res['r_med'])

    return sdflex


def tragus(video, ppm):
    mpDraw = mp.solutions.drawing_utils
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()

    cap = cv2.VideoCapture(video)
    pTime = 0

    # setting video output
    # out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, (int(cap.get(3)), int  (cap.get(4))))

    l_dist_all = []
    r_dist_all = []

    start = time.time()
    while cap.isOpened():
        success, img = cap.read()
        if success:
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(imgRGB)
            # results = pose.process(img)
            #print(results.pose_landmarks)
            points = []
            if results.pose_landmarks:
                mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    h, w, c = img.shape
                    print(id, lm)
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    points.append([cx, cy])
                    # cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            # cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

            # processing tragus
            r_dist_all.append(np.abs(points[12][0] - points[8][0]))
            l_dist_all.append(np.abs(points[11][0] - points[7][0]))


            # Write the frame into the file 'output.mp4'
            # cv2.imwrite('frame.jpg', img)

            # out.write(img)
            # cv2.imshow("Image", img)
            cv2.waitKey(1)
        else:
            break

    end = time.time()
    print("Analysis time: " + str(end - start) + " seconds")

    # Filter the results
    l_med = gb_utils.filterG(l_dist_all)
    r_med = gb_utils.filterG(r_dist_all)

    # STORE RESULTS TO DF

    data = pd.DataFrame({'l_med': l_med, 'r_med': r_med})

    data['l_med'] = data['l_med']/ppm
    data['r_med'] = data['r_med']/ppm

    res = data.min()
    print("res: ", res)

    tragus = Tragus(res['l_med'], res['r_med'])

    return tragus


def calib_func(videopath):
    # videopath: path to your video
    # chosen: the frame number (as an integer) you want to process in the video

    # Import required modules
    import cv2
    import numpy as np
    import os
    import glob
    from PIL import Image

    # Define the dimensions of the checkerboard
    CHECKERBOARD = (6, 8)
    scale = 0.025  # Distance in m between 2 corners of the checkerboard

    # stop the iteration when specified accuracy, epsilon, is reached or
    # specified number of iterations are completed.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Vector for 2D points
    a2dpoints = []

    # Edit next line to process specific frame of video
    chosenf = 60  # Frame no. used for calibration
    cap = cv2.VideoCapture(videopath)
    cap.set(1, chosenf)
    ret, frame = cap.read()

    # images = glob.glob('*.png')

    # for filename in images:
    # image = cv2.imread(filename)
    # grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayColor = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(grayColor, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH
                                             + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    dists = []
    # If desired number of corners are detected, refine the pixel coordinates and display
    if ret == True:
        # Refine pixel coordinates for 2d points.
        corners2 = cv2.cornerSubPix(grayColor, corners, (11, 11), (-1, -1), criteria)

        # a2dpoints.append(corners2)

        # Measure mean distance between a few corners on the checkerboard
        # This distance represents the 'scale' value
        for i in range(1, np.min(CHECKERBOARD)):
            x1 = corners2[i - 1][0][0]
            y1 = corners[i - 1][0][1]
            x2 = corners2[i][0][0]
            y2 = corners[i][0][1]
            dists.append(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))

        # print(str(scale) + ' m' + ' = ' + str(np.mean(dists)) + ' pixels')
        # print('----------')

        # Draw and display the corners, then save image (optional)
        frame = cv2.drawChessboardCorners(frame, CHECKERBOARD, corners2, ret)
        image2 = Image.fromarray(frame)
        image2.save(videopath[:-4] + '_calT' + '.png')

    # im_resize = cv2.resize(image, (960, 540))
    # cv2.imshow('img', im_resize)
    # cv2.waitKey(0)

    # cv2.destroyAllWindows()

    #return pixel per m
    ppm = (np.mean(dists))/scale

    print("PPM: ", ppm)
    
    if pd.isna(ppm):
        ppm = 0

    return ppm


moves_dict = {"sit2stand": sit2stand, "cerv_rot": cerv_rot, "hip_int_rot": hip_int_rot, "hip_abduction": hip_abduction,
              "lumbar_flex": lumbar_flex, "shoulder_flex": shoulder_flex, "side_flex": side_flex, "tragus": tragus}


def handle_move(move, ppm, path, email):
    process_move = moves_dict[move]
    global status
    status = 0
    try:
        res = process_move(path, ppm)
        res = res.__dict__
        status = add_movement_record(move, res, email)
    except Exception as e:
        print("Error in move process: ", str(e))
        # status = 0

    return status

# ppm = calib_func("./demo/nc_sit.mp4")
# ppm = sit2stand("./uploads/orefash1gmail.comsit2stand.mp4", 0.2, 'sit2stand', 'orefash1@gmail.com')
# ppm = hip_abduction("./uploads/orefash1gmail.comsit2stand.mp4", 0.2)
# #
# ppm = tragus("./demo/nc_sit.mp4", 0.2)
# print("PPM: ", ppm)
# print(ppm.__dict__)
# data = ppm.to_dict()

# cerv = CervRot(ppm['l_med'], ppm['r_med'])
# print(cerv.__dict__)

# import os.path
#
#
# file_path = "demo/cal.mp4"
# tet = os.path.isfile(file_path)
# print(tet)