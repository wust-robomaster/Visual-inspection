import pyrealsense2 as rs
import numpy as np
import cv2
from yolo import YOLO
import time
from PIL import Image

yolo = YOLO()
fps = 0.0

# 连接传感器----------------------------------------------------------------------------------------------------
pipeline = rs.pipeline()

# 配置传感器的参数----------------------------------------------------------------------------------------------------
config = rs.config()
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 60)
config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(config)

# 获取第一个连接的设备----------------------------------------------------------------------------------------------------
depth_sensor = profile.get_device().first_depth_sensor()

# 获取深度像素与真实距离的换算比例----------------------------------------------------------------------------------------------------
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)

frames = pipeline.wait_for_frames()
color = frames.get_color_frame()

# 获取颜色帧内参-------------------------------------------------------------------------------------------------------
color_profile = color.get_profile()
cvsprofile = rs.video_stream_profile(color_profile)
color_intrin = cvsprofile.get_intrinsics()
color_intrin_part = [color_intrin.ppx, color_intrin.ppy, color_intrin.fx, color_intrin.fy]
frames = pipeline.wait_for_frames()
color = frames.get_color_frame()

# 1米
clipping_distance_in_meters = 1
clipping_distance = clipping_distance_in_meters / depth_scale

# d2c 深度图对齐到彩色图
align_to = rs.stream.color
align = rs.align(align_to)
colorizer = rs.colorizer()

hole_filling = rs.hole_filling_filter()
# decimation = rs.decimation_filter()
# spatial = rs.spatial_filter()
# temporal = rs.temporal_filter()

try:
    while True:
        t1 = time.time()
        # 采集一帧数据，并且对齐图像
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()

        width = depth_frame.width
        height = depth_frame.height

        frame = depth_frame

        # frame = decimation.process(frame)
        # frame = spatial.process(frame)
        # frame = temporal.process(frame)
        frame = hole_filling.process(frame)

        depth_frame2 = np.asanyarray(colorizer.colorize(frame).get_data())
        color_frame = frames.get_color_frame()
        #
        # if not color_frame:
        #     continue
        color_frame = np.asanyarray(color_frame.get_data())
        # # 格式转变 BGR2RGB
        color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
        # # 转变成Image
        color_frame = Image.fromarray(np.uint8(color_frame))
        #
        # 进行检测
        color_frame = np.array(yolo.detect_image(color_frame, depth_frame, color_intrin_part, mode=1))
        color_frame = cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR)

        fps = (fps + (1. / (time.time() - t1))) / 2
        print("fps= %.2f" % (fps))
        # color_frame = cv2.putText(color_frame, "fps= %.2f" % (fps), (0, color_frame.shape[0] - 10),
        #                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("rgb", color_frame)
        # cv2.imwrite('color_frame15.jpg', color_frame)
        cv2.imshow('depth', depth_frame2)
        # cv2.imwrite('depth_frame.jpg', depth_frame2)
        # cv2.imshow('depth_color', depth_colormap)
        c = cv2.waitKey(1) & 0xff
        if c == 27:
            cv2.destoryAllWindow()
            pipeline.stop()
finally:
    # stop streaming
    pipeline.stop()