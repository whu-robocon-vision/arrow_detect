import pyrealsense2 as rs
import cv2 as cv
import sys
import numpy as np


def nothing(x):
    pass


def getdistance(p1, p2):
    return np.int0(np.sqrt(np.square(p1[0] - p2[0]) + np.square(p1[1] - p2[1])))


def get_extend_point(p1, p2, d1, d2, scale=5):
    px = np.int0((scale * d1 / d2) * (p1[0] - p2[0]) + p1[0])
    py = np.int0((scale * d1 / d2) * (p1[1] - p2[1]) + p1[1])
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
    if flag1:
        cv.drawContours(frame, [box1], 0, (255, 0, 255), 3)
    if flag2:
        cv.drawContours(frame, [box2], 0, (255, 0, 255), 3)
    return flag1 or flag2


def start_rs():
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
    cfg.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 60)
    cfg.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 60)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
    profile = pipeline.start(cfg)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    clipping_distance_in_meters = 1  # 1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    align_to = rs.stream.color
    align = rs.align(align_to)
    return pipeline, align


def create_window():
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


def detect_arrow(color_frame):
    frame = cv.cvtColor(color_frame, cv.COLOR_BGR2HSV)
    mask = cv.inRange(frame,
                        lowerb=np.array([cv.getTrackbarPos('Bl', 'param'), cv.getTrackbarPos(
                            'Gl', 'param'), cv.getTrackbarPos('Rl', 'param')]),
                        upperb=np.array([cv.getTrackbarPos('Bh', 'param'), cv.getTrackbarPos('Gh', 'param'), cv.getTrackbarPos('Rh', 'param')]))
    # mask = cv.bitwise_not(mask)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (cv.getTrackbarPos(
        'kernel', 'param'), cv.getTrackbarPos('kernel', 'param')))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=1)
    # mask = cv.Laplacian(mask, cv.CV_8U, ksize=5)
    _, contours, heriachy = cv.findContours(
        mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
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
        if (contour_area > cv.getTrackbarPos('factor', 'param') / 10 * w * h
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
                cv.circle(color_frame, (np.int(cx), np.int(cy)),
                            3, (0, 255, 255), -1)
    cv.imshow("color_frame", color_frame)


pipeline, align = start_rs()   
create_window()
while True:
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    if color_frame is None:
        continue
    color_frame = np.asanyarray(color_frame.get_data())
    detect_arrow(color_frame)
    if cv.waitKey(1) == 27:
        break
