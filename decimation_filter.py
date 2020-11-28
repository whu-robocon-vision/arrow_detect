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
# depth_frame:49892
# depth_frame:0.16240885416666667

# 【Visualising the Data 可视化数据】
# 创建着色器(其实这个可以替代opencv的convertScaleAbs()和applyColorMap()函数了,但是是在多少米范围内map呢?)
colorizer = rs.colorizer()

# 绘图不显示网格
plt.rcParams["axes.grid"] = False
# 图形尺寸,单位英尺
plt.rcParams['figure.figsize'] = [8, 4]

# 【Applying Filters 应用过滤器】
# [1、抽取过滤器]
# 抽取
# 使用立体深度解决方案时，z精度与原始空间分辨率有关。
# 如果您对较低的空间分辨率感到满意，则“抽取滤波器”将降低空间分辨率，以保持z精度并执行一些基本的孔填充。

# 创建抽取过滤器
decimation = rs.decimation_filter()
# decimated_depth = decimation.process(depth_frame)
# print(type(decimation)) # <class 'pyrealsense2.pyrealsense2.decimation_filter'>
# print(type(decimated_depth))  # <class 'pyrealsense2.pyrealsense2.frame'>

# # 您可以通过滤波器幅度选项来控制抽取量（线性比例因子）。
# # 注意不断变化的图像分辨率
decimation.set_option(rs.option.filter_magnitude, 4)
decimated_depth = decimation.process(depth_frame)
colorized_depth = np.asanyarray(colorizer.colorize(decimated_depth).get_data())
plt.imshow(colorized_depth)

# 【计算深度图数据中的0值】
num = 0
all = 0
for i in np.asanyarray(decimated_depth.get_data()).ravel():
    all += 1
    if i == 0:
        num += 1
print('decimated_depth分辨率：{}'.format(np.asanyarray(decimated_depth.get_data()).shape))
print('decimated_depth:{}'.format(num))
print('decimated_depth:{}'.format(num / all))
# decimated_depth分辨率：(240, 320)
# decimated_depth:10507
# decimated_depth:0.13680989583333333

plt.show()
