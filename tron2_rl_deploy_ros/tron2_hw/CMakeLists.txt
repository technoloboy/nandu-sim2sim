cmake_minimum_required(VERSION 3.10)
project(tron2_hw)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  geometry_msgs
  sensor_msgs
  std_msgs
  std_srvs
  controller_manager
  controller_manager_msgs
  hardware_interface
  robot_common
  realtime_tools
  urdf
  limxsdk_lowlevel
)

find_package(urdfdom REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS
    roscpp
    geometry_msgs
    sensor_msgs
    std_msgs
    std_srvs
    controller_manager
    controller_manager_msgs
    hardware_interface
    robot_common
    realtime_tools
    urdf
    limxsdk_lowlevel
)

include_directories(
  include
  ${CATKIN_DEVEL_PREFIX}/include
  ${catkin_INCLUDE_DIRS}
)

link_directories(${CATKIN_DEVEL_PREFIX}/lib)

add_executable(tron2_hw_node
  src/Tron2HW.cpp
  src/RobotHWLoop.cpp
  src/tron2_hw_node.cpp
)

add_dependencies(tron2_hw_node
  ${catkin_EXPORTED_TARGETS}
  limxsdk_lowlevel_target
)

target_link_libraries(tron2_hw_node
  ${catkin_LIBRARIES}
  ${urdfdom_LIBRARIES}
  limxsdk_lowlevel
  pthread
)

install(TARGETS tron2_hw_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/. DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})
install(DIRECTORY config launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
install(DIRECTORY docs DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
