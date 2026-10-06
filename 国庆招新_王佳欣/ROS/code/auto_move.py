#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
import time

def main():
    rospy.init_node("turtle_auto_node")
    pub = rospy.Publisher("/turtle1/cmd_vel",Twist,queue_size=10)
    rate = rospy.Rate(10)
    vel = Twist()

    #循环跑正方形：前进2秒，原地转90度，循环4次
    while not rospy.is_shutdown():
        for _ in range(4):
            #前进直线
            vel.linear.x = 0.5
            vel.angular.z = 0
            pub.publish(vel)
            time.sleep(2)
            #停止
            vel.linear.x =0
            pub.publish(vel)
            time.sleep(0.2)
            #原地旋转
            vel.linear.x = 0
            vel.angular.z = 0.8
            pub.publish(vel)
            time.sleep(1.6)
            #停止旋转
            vel.angular.z=0
            pub.publish(vel)
            time.sleep(0.2)

if __name__ == "__main__":
    main()
