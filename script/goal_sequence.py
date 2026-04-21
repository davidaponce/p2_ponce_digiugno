#!/usr/bin/env python3

import math
import rospy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from tf.transformations import quaternion_from_euler, euler_from_quaternion


class GoalSequenceNavigator:
    def __init__(self):
        rospy.init_node("goal_sequence_node")

        self.goal_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=10)
        rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.pose_callback)

        self.current_pose = {
            "x": None,
            "y": None,
            "yaw": None
        }

        # Replace these later with your actual measured map coordinates
        self.waypoints = [
            {"name": "Goal_A", "x": 1.92, "y": 1.45, "yaw": 0.76},
            {"name": "Goal_B", "x":  1.3447, "y": 4.5617, "yaw": 0.0117},
            {"name": "Goal_C", "x":  -0.3609, "y": 6.4622, "yaw": 0.0373},
        ]

        self.position_tolerance = rospy.get_param("~position_tolerance", 0.25)
        self.angle_tolerance = rospy.get_param("~angle_tolerance", 0.30)
        self.pause_time = rospy.get_param("~pause_time", 2.0)
        self.publish_retries = rospy.get_param("~publish_retries", 3)

        rospy.loginfo("Waiting for AMCL pose...")
        while not rospy.is_shutdown() and self.current_pose["x"] is None:
            rospy.sleep(0.2)

        rospy.loginfo("Pose received. Ready to start navigation.")

    def pose_callback(self, msg):
        pose = msg.pose.pose
        q = pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

        self.current_pose["x"] = pose.position.x
        self.current_pose["y"] = pose.position.y
        self.current_pose["yaw"] = yaw

    def normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def distance_to_goal(self, goal):
        dx = goal["x"] - self.current_pose["x"]
        dy = goal["y"] - self.current_pose["y"]
        return math.hypot(dx, dy)

    def heading_error(self, goal):
        error = goal["yaw"] - self.current_pose["yaw"]
        return abs(self.normalize_angle(error))

    def goal_reached(self, goal):
        if self.current_pose["x"] is None:
            return False

        distance_ok = self.distance_to_goal(goal) <= self.position_tolerance
        heading_ok = self.heading_error(goal) <= self.angle_tolerance
        return distance_ok and heading_ok

    def build_goal_message(self, goal):
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"

        msg.pose.position.x = goal["x"]
        msg.pose.position.y = goal["y"]
        msg.pose.position.z = 0.0

        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, goal["yaw"])
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw

        return msg

    def publish_goal(self, goal):
        msg = self.build_goal_message(goal)

        for attempt in range(self.publish_retries):
            msg.header.stamp = rospy.Time.now()
            self.goal_pub.publish(msg)
            rospy.loginfo(f"Published {goal['name']} (attempt {attempt + 1}/{self.publish_retries})")
            rospy.sleep(0.3)

    def navigate_to_goal(self, goal):
        rospy.loginfo(f"Navigating to {goal['name']}")
        self.publish_goal(goal)

        rate = rospy.Rate(5)

        while not rospy.is_shutdown():
            dist = self.distance_to_goal(goal)
            ang_err = self.heading_error(goal)

            rospy.loginfo_throttle(
                1.0,
                f"{goal['name']} -> distance: {dist:.2f}, heading error: {ang_err:.2f}"
            )

            if self.goal_reached(goal):
                rospy.loginfo(f"Reached {goal['name']}")
                rospy.sleep(self.pause_time)
                return True

            rate.sleep()

        return False

    def run(self):
        ordered_goals = [
            self.waypoints[1],
            self.waypoints[2],
            self.waypoints[0],
        ]

        for goal in ordered_goals:
            success = self.navigate_to_goal(goal)
            if not success:
                rospy.logwarn("Navigation interrupted before finishing route.")
                return

        rospy.loginfo("Completed full waypoint sequence.")


if __name__ == "__main__":
    try:
        navigator = GoalSequenceNavigator()
        navigator.run()
    except rospy.ROSInterruptException:
        pass
