姓名：王佳欣
学号：260341411
班级：26计科4班
QQ：2022176865
完成模块：ROS、OpenCV、YOLO

运行环境：
Ubuntu 20.04 + ROS Noetic + Python 3.8 + OpenCV
Windows11+ Ultralytics YOLO

ROS：
使用 auto_move.py 发布 /turtle1/cmd_vel 的 geometry_msgs/Twist 消息，
可以让小海龟自动画圆或正方形，start_turtle.launch 可一键启动。

OpenCV：
使用 HSV + inRange 完成颜色阈值；
使用 GrabCut 完成奶龙和蜡笔小新的分割；
使用 Haar 分类器完成人脸检测。

YOLO：
使用 LabelImg 将 46 张图片标注为 hulu 单类别数据集，
采用 yolo11n.pt 训练，并使用 best.pt 推理验证集。
