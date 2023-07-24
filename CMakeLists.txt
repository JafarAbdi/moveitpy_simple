cmake_minimum_required(VERSION 3.22)
project(moveitpy_simple)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS example/visualizer.py DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ros_testing REQUIRED)
  add_ros_test(moveitpy_simple/moveit_configs_utils/test/test_launch_utils.py
               TIMEOUT 120 ARGS robot_name:=test_robot robot_ip:=0.0.0.0)
endif()

ament_package()
