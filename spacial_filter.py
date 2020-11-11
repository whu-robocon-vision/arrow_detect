# -*- coding: utf-8 -*-
"""
@File    : test_测试深度过滤器_depth_filters.py
@Time    : 2019/12/17 11:29
@Author  : Dontla
@Email   : sxana@qq.com
@Software: PyCharm
"""

import numpy as np  # fundamental package for scientific computing 科学计算的基本软件包
import matplotlib.pyplot as plt  # 2D plotting library producing publication quality figures 2D绘图库产生出版物质量数据
import pyrealsense2 as rs  # Intel RealSense cross-platform open-source API 英特尔实感跨平台开源API

print("Environment Ready")

# 【Setup: 配置】
pipe = rs.pipeline()
cfg = rs.config()
# cfg.enable_device_from_file("stairs.bag")
# cfg.enable_device('838212073161')
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipe.start(cfg)

# 【Skip 5 first frames to give the Auto-Exposure time to adjust 跳过前5帧以设置自动曝光时间】
for x in range(5):
    pipe.wait_for_frames()

# 【Store next frameset for later processing: 存储下一个框架集以供以后处理：】
frameset = pipe.wait_for_frames()
depth_frame = frameset.get_depth_frame()

# 【Cleanup: 清理：】
pipe.stop()
print("Frames Captured")

# 【计算深度图数据中的0值】
num = 0
all = 0
for i in np.asanyarray(depth_frame.get_data()).ravel():
    all += 1
    if i == 0:
        num += 1
print('depth_frame分辨率：{}'.format(np.asanyarray(depth_frame.get_data()).shape))
print('depth_frame:{}'.format(num))
print('depth_frame:{}'.format(num / all))
# depth_frame分辨率：(480, 640)
# depth_frame:127369
# depth_frame:0.41461263020833333

# 【Visualising the Data 可视化数据】
# 创建着色器(其实这个可以替代opencv的convertScaleAbs()和applyColorMap()函数了,但是是在多少米范围内map呢?)
colorizer = rs.colorizer()
# colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
# print(colorized_depth.shape)  # (480, 640, 3)
# cv2.imshow('win', colorized_depth)
# cv2.waitKey(0)

# 绘图不显示网格
plt.rcParams["axes.grid"] = False
# 图形尺寸,单位英尺
plt.rcParams['figure.figsize'] = [8, 4]
# plt.imshow(colorized_depth)

# 【Applying Filters 应用过滤器】
# [2、空间过滤器]
# Spatial Filter
# Spatial Filter is a fast implementation of Domain-Transform Edge Preserving Smoothing
# 空间滤波器是域转换边缘保留平滑的快速实现
spatial = rs.spatial_filter()
# filtered_depth = spatial.process(depth_frame)

# We can emphesize the effect of the filter by cranking-up smooth_alpha and smooth_delta options:
# 我们可以通过增加smooth_alpha和smooth_delta选项来强调滤镜的效果：
spatial.set_option(rs.option.filter_magnitude, 5)
spatial.set_option(rs.option.filter_smooth_alpha, 1)
spatial.set_option(rs.option.filter_smooth_delta, 50)

# The filter also offers some basic spatial hole filling capabilities:
# 该过滤器还提供一些基本的空间孔填充功能：
spatial.set_option(rs.option.holes_fill, 3)

filtered_depth = spatial.process(depth_frame)
colorized_depth = np.asanyarray(colorizer.colorize(filtered_depth).get_data())
plt.imshow(colorized_depth)
plt.show()

# 【计算深度图数据中的0值】
num = 0
all = 0
for i in np.asanyarray(filtered_depth.get_data()).ravel():
    all += 1
    if i == 0:
        num += 1
print('filtered_depth分辨率：{}'.format(np.asanyarray(filtered_depth.get_data()).shape))
print('filtered_depth:{}'.format(num))
print('filtered_depth:{}'.format(num / all))
# filtered_depth分辨率：(480, 640)
# filtered_depth:29913
# filtered_depth:0.097373046875

