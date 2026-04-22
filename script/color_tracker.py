#!/usr/bin/env python3
#chmod +x ~/catkin_ws/src/p2_ponce_digiugno/scripts/color_tracker.py

import rospy
import cv2
import numpy as np

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist


class BallFollower:
    def __init__(self):
        rospy.init_node("ball_follower", anonymous=False)

        self.bridge = CvBridge()

        self.rgb_topic = rospy.get_param("~rgb_topic", "/camera/color/image_raw")
        self.depth_topic = rospy.get_param("~depth_topic", "/camera/depth/image_raw")
        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "/cmd_vel")

        # Desired behavior
        self.target_distance = rospy.get_param("~target_distance", 1.0)   # meters
        self.max_linear_speed = rospy.get_param("~max_linear_speed", 0.2)
        self.max_angular_speed = rospy.get_param("~max_angular_speed", 0.6)

        # Tuning
        self.distance_tolerance = rospy.get_param("~distance_tolerance", 0.1)
        self.center_tolerance = rospy.get_param("~center_tolerance", 40)  # pixels
        self.min_contour_area = rospy.get_param("~min_contour_area", 300)

        self.latest_bgr = None
        self.latest_depth = None

        self.cmd_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=10)

        self.rgb_sub = rospy.Subscriber(self.rgb_topic, Image, self.rgb_callback, queue_size=1)
        self.depth_sub = rospy.Subscriber(self.depth_topic, Image, self.depth_callback, queue_size=1)

        self.rate = rospy.Rate(10)

        rospy.loginfo("Ball follower started.")
        rospy.loginfo("RGB topic: %s", self.rgb_topic)
        rospy.loginfo("Depth topic: %s", self.depth_topic)

    def rgb_callback(self, msg):
        try:
            self.latest_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            rospy.logerr("RGB CvBridge error: %s", e)

    def depth_callback(self, msg):
        try:
            # Common depth encodings:
            # 32FC1 -> meters
            # 16UC1 -> usually millimeters
            rospy.loginfo_once("Depth shape: %s dtype: %s", self.latest_depth.shape, self.latest_depth.dtype)
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except CvBridgeError as e:
            rospy.logerr("Depth CvBridge error: %s", e)

    def detect_red_ball(self, bgr_image):
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

        # Red wraps around in HSV, so use two ranges
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])

        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 | mask2

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, None, mask

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area < self.min_contour_area:
            return None, None, mask

        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None, None, mask

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        return (cx, cy), largest, mask

    def get_depth_at_pixel(self, depth_image, x, y):
        if depth_image is None:
            return None

        h, w = depth_image.shape[:2]
        if x < 0 or x >= w or y < 0 or y >= h:
            return None

        # Average a small patch for stability
        patch_radius = 2
        xs = max(0, x - patch_radius)
        xe = min(w, x + patch_radius + 1)
        ys = max(0, y - patch_radius)
        ye = min(h, y + patch_radius + 1)

        patch = depth_image[ys:ye, xs:xe]

        if patch.size == 0:
            return None

        # Convert depending on type
        if patch.dtype == np.uint16:
            # Usually millimeters -> meters
            valid = patch[(patch > 0)]
            if valid.size == 0:
                return None
            return float(np.mean(valid)) / 1000.0
        else:
            # Usually float meters
            valid = patch[np.isfinite(patch)]
            valid = valid[valid > 0]
            if valid.size == 0:
                return None
            return float(np.mean(valid))

    def compute_cmd(self, ball_center, image_width, distance):
        cmd = Twist()

        if ball_center is None:
            return cmd

        cx, _ = ball_center
        image_center_x = image_width // 2
        x_error = cx - image_center_x

        # Turn toward the ball
        if abs(x_error) > self.center_tolerance:
            cmd.angular.z = -0.002 * x_error
            cmd.angular.z = max(-self.max_angular_speed,
                                min(self.max_angular_speed, cmd.angular.z))

        # Move to maintain desired distance
        if distance is not None:
            d_error = distance - self.target_distance

            if abs(d_error) > self.distance_tolerance:
                cmd.linear.x = 0.5 * d_error
                cmd.linear.x = max(-self.max_linear_speed,
                                   min(self.max_linear_speed, cmd.linear.x))

        return cmd

    def run(self):
        while not rospy.is_shutdown():
            if self.latest_bgr is None or self.latest_depth is None:
                self.rate.sleep()
                continue

            frame = self.latest_bgr.copy()
            ball_center, contour, mask = self.detect_red_ball(frame)

            distance = None
            if ball_center is not None:
                cx, cy = ball_center
                distance = self.get_depth_at_pixel(self.latest_depth, cx, cy)

                cv2.circle(frame, (cx, cy), 10, (255, 0, 0), 2)
                if contour is not None:
                    cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)

                label = "x={}, y={}".format(cx, cy)
                if distance is not None:
                    label += ", d={:.2f}m".format(distance)

                cv2.putText(frame, label, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cmd = self.compute_cmd(ball_center, frame.shape[1], distance)
            self.cmd_pub.publish(cmd)
            #debug movement
            rospy.loginfo("center=%s distance=%s lin=%.3f ang=%.3f",
              str(ball_center), str(distance), cmd.linear.x, cmd.angular.z)

            cv2.imshow("Ball Follower RGB", frame)
            cv2.imshow("Red Mask", mask if mask is not None else np.zeros((100, 100), dtype=np.uint8))
            cv2.waitKey(1)

            self.rate.sleep()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        node = BallFollower()
        node.run()
    except rospy.ROSInterruptException:
        pass