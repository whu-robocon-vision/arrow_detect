# -*- coding: utf-8 -*-
"""
@File    : test_191218_测试时间过滤器_Temporal_Filter.py
@Time    : 2019/12/18 10:59
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

# Our implementation of Temporal Filter does basic temporal smoothing and hole-filling.
# It is meaningless when applied to a single frame, so let's capture several consecutive frames:
# 我们的“时间过滤器”实现执行基本的时间平滑和孔填充。 当应用于单个帧时它是没有意义的，因此让我们捕获几个连续的帧：
frames = []
for x in range(10):
    frameset = pipe.wait_for_frames()
    frames.append(frameset.get_depth_frame())

pipe.stop()
print("Frames Captured")

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

# Next, we need to "feed" the frames to the filter one by one:
# 接下来，我们需要将帧逐一“馈入”到过滤器：
temporal = rs.temporal_filter()
for x in range(10):
    temp_filtered = temporal.process(frames[x])
    colorized_depth = np.asanyarray(colorizer.colorize(temp_filtered).get_data())
    plt.imshow(colorized_depth)
    plt.show()
# 您可以修改过滤器选项以微调结果（任何时间过滤都需要在平滑和运动之间进行权衡）

