import pyrealsense2 as rs
import cv2 as cv
import sys
import numpy as np

def nothing(x):
    pass

def getdistance(p1, p2):
    return np.int0(np.sqrt(np.square(p1[0] - p2[0]) + np.square(p1[1] - p2[1])))

def get_extend_point(p1, p2, d1, d2, scale = 5):
    px = np.int0((scale * d1 / d2 ) * (p1[0] - p2[0]) + p1[0])
    py = np.int0((scale * d1 / d2 ) * (p1[1] - p2[1]) + p1[1])
    return np.array([px, py])

def get_extend_box(box):
    d1 = getdistance(box[0], box[1])
    d2 = getdistance(box[1], box[2])
    scale = cv.getTrackbarPos('size', "param")
    if d1 < d2:
        p1 = get_extend_point(box[1], box[2], d1, d2, scale)
        p2 = get_extend_point(box[0], box[3], d1, d2, scale)
        box1 = np.array([box[0], box[1], p1, p2])
        p1 = get_extend_point(box[2], box[1], d1, d2, scale)
        p2 = get_extend_point(box[3], box[0], d1, d2, scale)
        box2 = np.array([box[2], p1, p2, box[3]])
        return box1, box2
    else:
        p1 = get_extend_point(box[1], box[0], d2, d1)
        p2 = get_extend_point(box[2], box[3], d2, d1)
        box1 = np.array([box[1], p1, p2, box[2]])
        p1 = get_extend_point(box[0], box[1], d2, d1)
        p2 = get_extend_point(box[3], box[2], d2, d1)
        box2 = np.array([box[0], box[3], p2, p1])
        return box1, box2

def check_arrow_head(frame, box):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mask = cv.fillPoly(mask, [box], 255)
    # cv.imshow("mask1", mask)
    mean = cv.mean(frame, mask)
    # print(mean)
    if (mean[0] > cv.getTrackbarPos('wbl', 'param') and mean[1] > cv.getTrackbarPos('wgl', 'param') and mean[2] > cv.getTrackbarPos('wrl', 'param')):
        return True
    else:
        return False

def check_head(box, frame):
    box1, box2 = get_extend_box(box)
    flag1 = check_arrow_head(frame, box1)
    flag2 = check_arrow_head(frame, box2)
    if not (flag1 ^ flag2):
        return False
    if flag1:
        cv.drawContours(frame, [box1], 0, (255, 0, 255), 3)
    if flag2:
        cv.drawContours(frame, [box2], 0, (255, 0, 255), 3)
    return True

def extrace_object_demo():
    cap = cv.VideoCapture(0)
    cv.namedWindow("param")
    cv.resizeWindow("param", 400, 800)
    cv.createTrackbar('Rl', "param", 0, 255, nothing)
    cv.createTrackbar('Rh', 'param', 38, 255, nothing)
    cv.createTrackbar('Gl', 'param', 0, 255, nothing)
    cv.createTrackbar('Gh', 'param', 190, 255, nothing)
    cv.createTrackbar('Bl', 'param', 0, 255, nothing)
    cv.createTrackbar('Bh', 'param', 180, 255, nothing)
    cv.createTrackbar('factor', 'param', 0, 10, nothing)
    cv.createTrackbar('area_min', 'param', 203, 3000, nothing)
    cv.createTrackbar('area_max', 'param', 6000, 12000, nothing)
    cv.createTrackbar('scale', 'param', 8, 30, nothing)
    cv.createTrackbar('kernel', 'param', 3, 10, nothing)
    cv.createTrackbar('size', 'param', 3, 15, nothing)
    cv.createTrackbar('wrl', 'param', 117, 255, nothing)
    cv.createTrackbar('wgl', 'param', 110, 255, nothing)
    cv.createTrackbar('wbl', 'param', 110, 255, nothing)

    while True:
        _, color_frame = cap.read()
        frame = color_frame
        # frame = cv.GaussianBlur(color_frame, (7, 7), 0)
        # cv.imshow("blurred", frame)
        frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        mask = cv.inRange(frame,
            lowerb=np.array([cv.getTrackbarPos('Bl', 'param'), cv.getTrackbarPos('Gl', 'param'), cv.getTrackbarPos('Rl', 'param')]), 
            upperb=np.array([cv.getTrackbarPos('Bh', 'param'), cv.getTrackbarPos('Gh', 'param'), cv.getTrackbarPos('Rh', 'param')]))
        # mask = cv.bitwise_not(mask)
        cv.imshow("range", mask)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (cv.getTrackbarPos('kernel', 'param'), cv.getTrackbarPos('kernel', 'param')))
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=1)

        # mask = cv.Laplacian(mask, cv.CV_8U, ksize=5)
        _, contours, heriachy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for i, contour in enumerate(contours):

            # epsilon = 0.1 * cv.arcLength(contour, True)
            # contour = cv.approxPolyDP(contour, epsilon, True)

            # contour = cv.convexHull(contour)

            contour_area = cv.contourArea(contour)
            rect = cv.minAreaRect(contour)
            w, h = rect[1]
            # cv.drawContours(frame, contours, i, (255, 0, 255), 3)
            if min(w, h) == 0:
                scale = 0
            else:
                scale = max(w, h) / min(w, h)
            if  (contour_area > cv.getTrackbarPos('factor', 'param') / 10 * w * h 
                and contour_area > cv.getTrackbarPos('area_min', 'param')
                and contour_area < cv.getTrackbarPos('area_max', 'param')
                and scale > cv.getTrackbarPos('scale', 'param')):
                
                box = cv.boxPoints(rect)
                box = np.int0(box)
                if check_head(box, color_frame):
                    cv.drawContours(color_frame, [box], 0, (0, 0, 255), 3)
                    mm = cv.moments(contour)
                    cx = mm['m10'] / mm['m00']
                    cy = mm['m01'] / mm['m00']
                    cv.circle(color_frame, (np.int(cx), np.int(cy)), 3, (0, 255, 255), -1)
                
        cv.imshow("mask", mask)
        cv.imshow("frame", color_frame)

        c = cv.waitKey(1)
        if c == 27:
            cv.destroyAllWindows()
            break


#     while(True):
#         frames = pipeline.wait_for_frames()
#         frame = frames.get_color_frame()


#         # ret, frame = capture.read()
#         # if ret == False:
#         #     break

#         # lower_hsv = np.array([156, 60, 46])
#         # upper_hsv = np.array([180, 255, 255])
#         # frame = cv.pyrMeanShiftFiltering(frame, 20, 100)
#         frame = cv.GaussianBlur(frame, (3, 3), 0)
#         # hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
#         # mask1 = cv.inRange(hsv, lowerb=lower_hsv, upperb=upper_hsv)
#         mask = cv.inRange(frame, lowerb=np.array([0, 0, 110]), upperb=np.array([110, 110, 255]))
#         kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
#         mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
#         # mask = cv.bitwise_or(mask1, mask2)

        
#         # dst = cv.bitwise_and(frame, frame, mask=mask)

#         contours, heriachy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#         for i, contour in enumerate(contours):
#             contour_area = cv.contourArea(contour)
#             x, y, w, h = cv.boundingRect(contour)
#             cv.drawContours(mask, contours, i, (0, 0, 255), 3)
#             if contour_area > 0.8 * w * h:
#                 cv.drawContours(frame, contours, i, (255, 0, 255), 2)

#         cv.imshow("mask", mask)
#         cv.imshow("video", frame)
        
#         if cv.waitKey(40) == 27:
#             break

t1 = cv.getTickCount()
extrace_object_demo()
t2 = cv.getTickCount()
print("time: %s ms"%((t2 - t1) / cv.getTickFrequency() * 1000))

# cv.destroyAllWindows()