import unittest

import launch_testing.actions
import pytest
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch_testing.util import KeepAliveProc

from moveitpy_simple.moveit_configs_utils.launch_utils import launch_configurations


@launch_configurations
def launch_descriptions(launch_configurations):
    assert launch_configurations.robot_name == "test_robot"
    assert launch_configurations.robot_ip == "0.0.0.0"
    assert launch_configurations.robot_length == "2"
    with pytest.raises(AttributeError):
        launch_configurations.robot_height


def generate_test_description():
    ls_executable = ExecuteProcess(
        cmd=["ls", "-l"],
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("robot_name"),
            DeclareLaunchArgument("robot_ip"),
            DeclareLaunchArgument("robot_length", default_value="2"),
            ls_executable,
            KeepAliveProc(),
            launch_testing.actions.ReadyToTest(),
            *launch_descriptions(),
        ],
    ), {"test_executable": ls_executable}


class TestWaitForCompletion(unittest.TestCase):
    # Waits for test to complete, then waits a bit to make sure result files are generated
    def test_gtest_run_complete(self, test_executable):
        self.proc_info.assertWaitForShutdown(test_executable, timeout=4000.0)


@launch_testing.post_shutdown_test()
class TestProcessPostShutdown(unittest.TestCase):
    # Checks if the test has been completed with acceptable exit codes
    def test_gtest_pass(self, proc_info, test_executable):
        launch_testing.asserts.assertExitCodes(proc_info, process=test_executable)
