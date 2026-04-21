cmake_minimum_required(VERSION 3.0.2)
project(p2_ponce_digiugno)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  sensor_msgs
  tf
  cv_bridge
  image_transport
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)
