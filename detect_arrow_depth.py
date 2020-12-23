import pyrealsense2 as rs
import cv2 as cv
import sys
import numpy as np


def nothing(x):
    pass


def extrace_object_demo():
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipe.start(cfg)

    colorizer = rs.colorizer()

    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)

    decimation = rs.decimation_filter()
    decimation.set_option(rs.option.filter_magnitude, 4)

    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 5)
    spatial.set_option(rs.option.filter_smooth_alpha, 1)
    spatial.set_option(rs.option.filter_smooth_delta, 50)
    spatial.set_option(rs.option.holes_fill, 3)

    temporal = rs.temporal_filter()

    hole_filling = rs.hole_filling_filter()

    # capture = cv.VideoCapture(0)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    clipping_distance_in_meters = 1  # 1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    align_to = rs.stream.color
    align = rs.align(align_to)

    cv.namedWindow("frame")
    cv.createTrackbar('max_distance', "frame", 3000, 10000, nothing)
    cv.createTrackbar('min_distance', 'frame', 300, 3000, nothing)
    cv.createTrackbar('kernel', 'frame', 1, 10, nothing)
    cv.createTrackbar('thresh', 'frame', 120, 255, nothing)
    cv.createTrackbar('factor', 'frame', 0, 10, nothing)
    cv.createTrackbar('area_min', 'frame', 300, 3000, nothing)
    cv.createTrackbar('area_max', 'frame', 16000, 30000, nothing)
    cv.createTrackbar('scale', 'frame', 1, 30, nothing)

    while True:
        frames = pipe.wait_for_frames()
        aligned_frames = align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        frame = aligned_frames.get_depth_frame()
        # frame = decimation.process(frame)
        # frame = depth_to_disparity.process(frame)
        # frame = spatial.process(frame)
        # frame = temporal.process(frame)
        # frame = disparity_to_depth.process(frame)
        # frame = hole_filling.process(frame)

        depth_frame = np.asanyarray(frame.get_data())
        color_frame = np.asanyarray(color_frame.get_data())
        colorized_depth = np.asanyarray(colorizer.colorize(frame).get_data())

        _, depth_frame = cv.threshold(depth_frame,
                                      cv.getTrackbarPos('max_distance', 'frame'), 255, cv.THRESH_TOZERO_INV)
        _, depth_frame = cv.threshold(depth_frame,
                                      cv.getTrackbarPos('min_distance', 'frame'), 255, cv.THRESH_TOZERO)
        depth_frame = depth_frame * 255.0 / \
            (cv.getTrackbarPos('max_distance', 'frame') - cv.getTrackbarPos('min_distance', 'frame')) - \
            cv.getTrackbarPos('min_distance', 'frame') / \
            (cv.getTrackbarPos('max_distance', 'frame') - cv.getTrackbarPos('min_distance', 'frame')) 

        depth_frame = np.uint8(depth_frame)

        kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        depth_frame = cv.dilate(depth_frame, kernel)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        depth_frame = cv.erode(depth_frame, kernel)
        # cv.morphologyEx(depth_frame, cv.MORPH_OPEN, kernel, iterations=3)
        # depth_frame = cv.Laplacian(depth_frame, cv.CV_8U, ksize=5)
        depth_frame = cv.Canny(depth_frame, 50, 150)
        cv.imshow('laplacian', depth_frame)

        _, depth_frame = cv.threshold(depth_frame, cv.getTrackbarPos(
            'thresh', 'frame'), 255, cv.THRESH_BINARY)

        
        # depth_frame = cv.resize(depth_frame, (640, 480))

        _, contours, heriachy = cv.findContours(
            depth_frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        for i, contour in enumerate(contours):
            
            contour = cv.convexHull(contour)

            # cv.drawContours(color_frame, contours, i, (0, 255, 255), 3)
            # rect = cv.minAreaRect(contour)
            
            if (len(contour) > 5):
                rect = cv.fitEllipse(contour)
            else:
                continue
            contour_area = cv.contourArea(contour)
            w, h = rect[1]
            if min(w, h) == 0:
                scale = 0
            else:
                scale = max(w, h) / min(w, h)
            if ((contour_area > (cv.getTrackbarPos('factor', 'frame') / 10 * w * h))
                and (contour_area > cv.getTrackbarPos('area_min', 'frame'))
                and (contour_area < cv.getTrackbarPos('area_max', 'frame'))
                    and (scale > cv.getTrackbarPos('scale', 'frame'))):

                box = cv.boxPoints(rect)
                box = np.int0(box) * 4
                cv.drawContours(color_frame, [box], 0, (255, 255, 0), 3)
                mm = cv.moments(contour)
                cx = mm['m10'] / mm['m00']
                cy = mm['m01'] / mm['m00']
                cv.circle(color_frame, (np.int(cx) * 4, np.int(cy) * 4),
                          3, (255, 0, 255), -1)
                cv.circle(depth_frame, (np.int(cx), np.int(cy)),
                          5, (255, 0, 255), -1)
            else:
                continue
                box = cv.boxPoints(rect)
                box = np.int0(box)
                cv.drawContours(color_frame, [box], 0, (255, 0, 255), 3)

        cv.imshow("frame", depth_frame)
        cv.imshow("color_frame", color_frame)
        cv.imshow('color_depth', colorized_depth)

        c = cv.waitKey(1)
        if c == 27:
            cv.destroyAllWindows()
            break


t1 = cv.getTickCount()
extrace_object_demo()
t2 = cv.getTickCount()
print("time: %s ms" % ((t2 - t1) / cv.getTickFrequency() * 1000))
