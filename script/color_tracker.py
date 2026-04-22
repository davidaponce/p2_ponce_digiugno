#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

TOPIC_RGB   = "/camera/color/image_raw"
TOPIC_DEPTH = "/camera/depth/image_raw"
TOPIC_CMD   = "/cmd_vel"

TARGET_DIST = 1.0   # meters
LINEAR_GAIN = 0.4   # how fast to approach/retreat
ANGULAR_GAIN = 0.003 # how fast to turn toward ball
MAX_LINEAR  = 0.3   # m/s
MAX_ANGULAR = 1.0   # rad/s
MIN_AREA    = 500   # ignore tiny red blobs

# Red spans two HSV ranges (wraps around 0/180)
RED_RANGES = [
    (np.array([0,   120, 70]), np.array([10,  255, 255])),
    (np.array([170, 120, 70]), np.array([180, 255, 255])),
]

bridge = CvBridge()
rgb   = None
depth = None

def rgb_cb(msg):
    global rgb
    rgb = bridge.imgmsg_to_cv2(msg, "bgr8")

def depth_cb(msg):
    global depth
    depth = bridge.imgmsg_to_cv2(msg, "passthrough")

def find_ball(img):
    """Return (cx, cy) of the largest red blob, or None."""
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = sum(cv2.inRange(hsv, lo, hi) for lo, hi in RED_RANGES)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(biggest) < MIN_AREA:
        return None
    M = cv2.moments(biggest)
    return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

def get_depth_at(x, y, radius=5):
    """Return median depth (meters) in a small patch, or None if unavailable."""
    if depth is None:
        return None
    h, w = depth.shape[:2]
    patch = depth[max(0,y-radius):min(h,y+radius),
                  max(0,x-radius):min(w,x+radius)].astype(float)
    if depth.dtype == np.uint16:
        patch /= 1000.0
    valid = patch[(patch > 0) & ~np.isnan(patch)]
    return float(np.median(valid)) if valid.size else None

def clamp(val, limit):
    return max(-limit, min(limit, val))

def main():
    rospy.init_node("ball_follower")
    rospy.Subscriber(TOPIC_RGB,   Image, rgb_cb,   queue_size=1)
    rospy.Subscriber(TOPIC_DEPTH, Image, depth_cb, queue_size=1)
    cmd_pub = rospy.Publisher(TOPIC_CMD, Twist, queue_size=1)

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        if rgb is None:
            rate.sleep()
            continue

        frame  = rgb.copy()
        center = find_ball(frame)
        cmd    = Twist()

        if center is not None:
            cx, cy = center
            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), 2)

            # Turn to center the ball horizontally
            error_x = cx - frame.shape[1] // 2
            cmd.angular.z = clamp(-ANGULAR_GAIN * error_x, MAX_ANGULAR)

            # Drive to maintain target distance
            dist = get_depth_at(cx, cy)
            if dist is not None:
                cmd.linear.x = clamp(LINEAR_GAIN * (dist - TARGET_DIST), MAX_LINEAR)
                cv2.putText(frame, f"{dist:.2f}m", (cx + 15, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cmd_pub.publish(cmd)
        cv2.imshow("ball follower", frame)
        cv2.waitKey(1)
        rate.sleep()

if __name__ == "__main__":
    main()