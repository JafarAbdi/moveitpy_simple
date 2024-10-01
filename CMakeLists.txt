cmake_minimum_required(VERSION 3.22)
project(moveitpy_simple)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  find_package(ros_testing REQUIRED)
  add_ros_test(moveitpy_simple/moveit_configs_utils/test/test_launch_utils.py
               TIMEOUT 120 ARGS robot_name:=test_robot robot_ip:=0.0.0.0)
  ament_add_pytest_test(test_moveitpy
                        moveitpy_simple/moveitpy/test/test_moveitpy.py)
  ament_add_pytest_test(
    test_file_loaders
    moveitpy_simple/moveit_configs_utils/test/test_file_loaders.py)
  ament_add_pytest_test(
    test_moveit_resources_configs
    moveitpy_simple/moveit_configs_utils/test/test_moveit_resources_configs.py)
endif()

ament_package()
