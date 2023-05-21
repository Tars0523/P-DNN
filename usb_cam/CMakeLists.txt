cmake_minimum_required(VERSION 3.0.2)
project(usb_cam)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  message_generation
)
find_package(OpenCV REQUIRED)
generate_messages(DEPENDENCIES std_msgs)

catkin_package(
   CATKIN_DEPENDS rospy std_msgs
   #DEPENDS system_lib
)

catkin_install_python(PROGRAMS
    src/opencv-capture.py
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
