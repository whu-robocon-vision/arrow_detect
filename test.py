
import numpy as np  # fundamental package for scientific computing 科学计算的基本软件包
# Intel RealSense cross-platform open-source API 英特尔实感跨平台开源API
import pyrealsense2 as rs
import cv2 as cv

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

while True:
    frames = pipe.wait_for_frames()
    color_frame = frames.get_color_frame()
    frame = frames.get_depth_frame()
    frame = decimation.process(frame)
    frame = depth_to_disparity.process(frame)
    # frame = spatial.process(frame)
    # frame = temporal.process(frame)
    frame = disparity_to_depth.process(frame)
    frame = hole_filling.process(frame)

    depth_frame = np.asanyarray(frame.get_data())
    color_frame = np.asanyarray(color_frame.get_data())
    colorized_depth = np.asanyarray(colorizer.colorize(frame).get_data())
    cv.imshow('colorized_depth', colorized_depth)
    cv.imshow('depth_frame', depth_frame)
    cv.imshow('color_frame', color_frame)


    if cv.waitKey(1) == 27:
        break
pipe.stop()
