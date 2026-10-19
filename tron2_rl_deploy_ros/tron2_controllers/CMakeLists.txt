cmake_minimum_required(VERSION 3.10)
project(tron2_controllers)

set(CMAKE_CXX_STANDARD 17)
find_package(Eigen3 REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(orocos_kdl REQUIRED)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(catkin REQUIRED
  COMPONENTS
  roscpp
  robot_common
  robot_description
  controller_interface
  hardware_interface
  std_msgs
  geometry_msgs
  sensor_msgs
  diagnostic_msgs
  message_generation
  kdl_parser
  urdf
  onnxruntime_sdk
)

add_message_files(
  FILES
  WholeBodyPoseCommand.msg
  EEMotionCommand.msg
  NANDU1Command.msg
  NANDU1V14Command.msg
  NANDU1TCPCommand.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS
  roscpp
  robot_common
  robot_description
  controller_interface
  hardware_interface
  std_msgs
  geometry_msgs
  diagnostic_msgs
  message_runtime
  sensor_msgs
  kdl_parser
  urdf
  onnxruntime_sdk
  DEPENDS orocos_kdl
)

set(ONNXRUNTIME_SDK_LIB_FILES libonnxruntime.so libonnxruntime.so.1.10.0)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${orocos_kdl_INCLUDE_DIRS}
)

link_directories(${CATKIN_DEVEL_PREFIX}/lib)

add_library(${PROJECT_NAME} SHARED
  src/ControllerBase.cpp
  src/SolefootController.cpp
  src/WheelfootController.cpp
  src/WFYGController.cpp
  src/NANDU1Controller.cpp
  src/NANDU1AdaptiveController.cpp
)

add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${ONNXRUNTIME_SDK_LIB_FILES}
  OpenSSL::Crypto
  ${orocos_kdl_LIBRARIES}
)

if(CATKIN_ENABLE_TESTING)
  add_executable(nandu1_adaptive_profiles_test test/nandu1_adaptive_profiles_test.cpp)
  target_compile_options(nandu1_adaptive_profiles_test PRIVATE -UNDEBUG)
  add_test(NAME nandu1_adaptive_profiles COMMAND nandu1_adaptive_profiles_test)
  add_executable(nandu1_adaptive_onnx_conformance test/nandu1_adaptive_onnx_conformance.cpp)
  target_link_libraries(nandu1_adaptive_onnx_conformance libonnxruntime.so)
  add_executable(nandu1_adaptive_runtime_conformance test/nandu1_adaptive_runtime_conformance.cpp)
  target_link_libraries(nandu1_adaptive_runtime_conformance libonnxruntime.so)
  get_filename_component(NANDU1_ADAPTIVE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}" REALPATH)
  get_filename_component(NANDU1_ADAPTIVE_ROOT "${NANDU1_ADAPTIVE_ROOT}/../.." ABSOLUTE)
  set(NANDU1_ADAPTIVE_BUNDLE "${NANDU1_ADAPTIVE_ROOT}/deployment_bundles/nandu1_v1_3_1_2_6_model_23500")
  if(EXISTS "${NANDU1_ADAPTIVE_BUNDLE}/runtime_conformance.txt")
    add_test(NAME nandu1_adaptive_onnx COMMAND nandu1_adaptive_onnx_conformance "${NANDU1_ADAPTIVE_BUNDLE}")
    add_test(NAME nandu1_adaptive_runtime COMMAND nandu1_adaptive_runtime_conformance "${NANDU1_ADAPTIVE_BUNDLE}")
    add_test(NAME nandu1_adaptive_validator COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/scripts/validate_nandu1_bundle.py --bundle "${NANDU1_ADAPTIVE_BUNDLE}")
  endif()
  set(NANDU1_BALANCE_BUNDLE "${NANDU1_ADAPTIVE_ROOT}/deployment_bundles/nandu1_v1_3_1_2_2_model_29500")
  foreach(NANDU1_CASE IN ITEMS "3;11000" "5;10500" "7;10000")
    list(GET NANDU1_CASE 0 NANDU1_SUFFIX)
    list(GET NANDU1_CASE 1 NANDU1_ITERATION)
    set(NANDU1_CASE_BUNDLE "${NANDU1_ADAPTIVE_ROOT}/deployment_bundles/nandu1_v1_3_1_2_${NANDU1_SUFFIX}_model_${NANDU1_ITERATION}")
    if(EXISTS "${NANDU1_CASE_BUNDLE}/runtime_conformance.txt")
      add_test(NAME nandu1_ablation_${NANDU1_SUFFIX}_onnx COMMAND nandu1_adaptive_onnx_conformance "${NANDU1_CASE_BUNDLE}" 71)
      add_test(NAME nandu1_ablation_${NANDU1_SUFFIX}_runtime COMMAND nandu1_adaptive_runtime_conformance "${NANDU1_CASE_BUNDLE}" "Isaac-NANDU1-Blind-Flat-v1-3-1-2-${NANDU1_SUFFIX}")
      add_test(NAME nandu1_ablation_${NANDU1_SUFFIX}_validator COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/scripts/validate_nandu1_bundle.py --bundle "${NANDU1_CASE_BUNDLE}")
    endif()
  endforeach()
  if(EXISTS "${NANDU1_BALANCE_BUNDLE}/runtime_conformance.txt")
    add_test(NAME nandu1_adaptive_balance_onnx COMMAND nandu1_adaptive_onnx_conformance "${NANDU1_BALANCE_BUNDLE}" 70)
    add_test(NAME nandu1_adaptive_balance_runtime COMMAND nandu1_adaptive_runtime_conformance "${NANDU1_BALANCE_BUNDLE}" 70)
    add_test(NAME nandu1_adaptive_balance_validator COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/scripts/validate_nandu1_bundle.py --bundle "${NANDU1_BALANCE_BUNDLE}")
  endif()
  add_executable(nandu1_profiles_test test/nandu1_profiles_test.cpp)
  target_compile_options(nandu1_profiles_test PRIVATE -UNDEBUG)
  add_test(NAME nandu1_profiles COMMAND nandu1_profiles_test)
  add_executable(nandu1_runtime_test test/nandu1_runtime_test.cpp)
  target_compile_options(nandu1_runtime_test PRIVATE -UNDEBUG)
  add_test(NAME nandu1_runtime COMMAND nandu1_runtime_test)
  add_executable(nandu1_onnx_conformance test/nandu1_onnx_conformance.cpp)
  add_executable(nandu1_v14_conformance test/nandu1_v14_conformance.cpp)
  add_executable(nandu1_v131_conformance test/nandu1_v131_conformance.cpp)
  add_executable(nandu1_v141_conformance test/nandu1_v141_conformance.cpp)
  get_filename_component(NANDU1_CONTROLLER_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}" REALPATH)
  get_filename_component(NANDU1_DEPLOY_ROOT "${NANDU1_CONTROLLER_SOURCE}/../.." ABSOLUTE)
  set(NANDU1_V14_BUNDLE "${NANDU1_DEPLOY_ROOT}/deployment_bundles/nandu1_v1_4_model_24000")
  if(EXISTS "${NANDU1_V14_BUNDLE}/runtime_conformance.txt")
    add_test(NAME nandu1_v14_runtime COMMAND nandu1_v14_conformance "${NANDU1_V14_BUNDLE}")
    add_test(NAME nandu1_v14_onnx COMMAND nandu1_onnx_conformance "${NANDU1_V14_BUNDLE}")
  endif()
  target_link_libraries(nandu1_onnx_conformance libonnxruntime.so)
  foreach(bundle IN ITEMS nandu1_v1_2_model_40000 nandu1_v1_1_model_30000 nandu1_v1_3_model_30000 nandu1_v1_2_1_model_29000 nandu1_v1_3_1_model_30000 nandu1_v1_4_1_model_14000 nandu1_v1_3_1_2_model_27500)
    set(path "${NANDU1_DEPLOY_ROOT}/deployment_bundles/${bundle}")
    if(EXISTS "${path}/conformance.txt")
      add_test(NAME ${bundle}_onnx COMMAND nandu1_onnx_conformance "${path}")
      add_test(NAME ${bundle}_validator COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/scripts/validate_nandu1_bundle.py --bundle "${path}")
    endif()
  endforeach()
  if(EXISTS "${NANDU1_DEPLOY_ROOT}/deployment_bundles/nandu1_v1_3_1_model_30000/yaw_conformance.txt")
    add_test(NAME nandu1_v131_runtime COMMAND nandu1_v131_conformance "${NANDU1_DEPLOY_ROOT}/deployment_bundles/nandu1_v1_3_1_model_30000")
  endif()
  if(EXISTS "${NANDU1_DEPLOY_ROOT}/deployment_bundles/nandu1_v1_4_1_model_14000/runtime_conformance.txt")
    add_test(NAME nandu1_v141_runtime COMMAND nandu1_v141_conformance "${NANDU1_DEPLOY_ROOT}/deployment_bundles/nandu1_v1_4_1_model_14000/runtime_conformance.txt")
  endif()
  catkin_add_gtest(wfyg_math_test test/WFYGMathTest.cpp)
  if(TARGET wfyg_math_test)
    target_link_libraries(wfyg_math_test ${catkin_LIBRARIES} gtest_main)
  endif()
  catkin_add_nosetests(test/test_wfyg_math.py)
  catkin_add_nosetests(test/test_bundle_validator.py)
  add_test(NAME nandu1_container COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/test/test_nandu1_container.py)
  add_test(NAME nandu1_ros_params COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/test/test_nandu1_ros_params.py)
  add_test(NAME nandu1_keyboard COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/test/test_nandu1_keyboard.py)
  if(EXISTS "${NANDU1_V14_BUNDLE}/manifest.yaml")
    add_test(NAME nandu1_v14_launch COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/test/test_nandu1_v14_launch.py)
  endif()
endif()

install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/. DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})
install(DIRECTORY config launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
install(PROGRAMS scripts/validate_wfyg_bundle.py scripts/wfyg_motion_demo.py scripts/nandu1_adaptive_telemetry.py
  scripts/validate_nandu1_bundle.py scripts/nandu1_aux_receiver.py scripts/nandu1_command.py scripts/nandu1_mouse_bridge.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(FILES tron2_controllers_plugins.xml DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
