# Copyright information
#
# © [2024] LimX Dynamics Technology Co., Ltd. All rights reserved.

# Declare the minimum required CMake version
cmake_minimum_required(VERSION 3.10)

# Declare the project
project(robot_common)

find_package(catkin REQUIRED COMPONENTS hardware_interface)

catkin_package(INCLUDE_DIRS include CATKIN_DEPENDS hardware_interface)

# Install the include directory to the global Catkin include destination
install(DIRECTORY include/. DESTINATION ${CATKIN_GLOBAL_INCLUDE_DESTINATION})
