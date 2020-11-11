# -*- coding: utf-8 -*-
"""
@File    : test_191218_测试孔填充过滤器.py
@Time    : 2019/12/18 13:32
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

# 孔填充过滤器提供了附加的深度外推层：
hole_filling = rs.hole_filling_filter()
filled_depth = hole_filling.process(depth_frame)
colorized_depth = np.asanyarray(colorizer.colorize(filled_depth).get_data())
plt.imshow(colorized_depth)
plt.show()

