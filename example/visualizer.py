#!/usr/bin/env python3

"""A simple example of using moveitpy with panda3d_viewer and wandb."""

from copy import deepcopy
from pathlib import Path

from ament_index_python.packages import get_package_share_path
from moveit.core.robot_model import RobotModel
from moveit.core.robot_state import RobotState

import wandb
from moveitpy_simple.moveit_configs_utils import MoveItConfigsBuilder
from moveitpy_simple.moveit_configs_utils.file_loaders import load_xacro
from moveitpy_simple.moveitpy import MoveItPySimple
from moveitpy_simple.moveitpy.visualization import Visualizer

dir_path = Path(__file__).parent.absolute()


# TODO: This won't work for an installed package when run from ros2 run ...
moveit_configs = (
    MoveItConfigsBuilder(
        package=dir_path.parent
        / "moveitpy_simple"
        / "moveitpy"
        / "test"
        / "panda_moveit_config",
    )
    .load_all()
    .to_moveit_configs()
)


# Using MoveItPySimple for planning
moveitpy = MoveItPySimple(
    "moveitpy_simple",
    moveit_configs,
)


robot_state = RobotState(moveitpy.robot_model)
robot_state.set_to_default_values(moveitpy.arm.joint_model_group, "ready")
robot_state.set_to_default_values(moveitpy.gripper.joint_model_group, "open")
robot_state.update()

visualizer = Visualizer(
    moveit_configs.robot_description["robot_description"], "onscreen",
)
visualizer.visualize_robot_state(robot_state)

moveitpy.arm.set_start_state(robot_state)
moveitpy.arm.set_goal_from_named_state("extended")
plan = moveitpy.arm.plan()
if plan:
    visualizer.visualize_robot_trajectory(plan.trajectory)

# Without MoveItPySimple by creating a robot state with a robot model
#
urdf_file = load_xacro(
    get_package_share_path("moveit_resources_panda_moveit_config")
    / "config"
    / "panda.urdf.xacro",
)
srdf_file = load_xacro(
    get_package_share_path("moveit_resources_panda_moveit_config")
    / "config"
    / "panda.srdf",
)

robot_model = RobotModel(urdf_xml_string=urdf_file, srdf_xml_string=srdf_file)
robot_state = RobotState(robot_model)
robot_state.set_to_default_values("panda_arm", "ready")
robot_state.set_to_default_values("hand", "open")
robot_state.update()

robot_trajectory = [deepcopy(robot_state)]

robot_state.set_to_default_values("panda_arm", "extended")
robot_state.set_to_default_values("hand", "close")
robot_state.update()

robot_trajectory.append(deepcopy(robot_state))


visualizer.visualize_robot_state(robot_state)
visualizer.visualize_robot_trajectory(robot_trajectory)

wandb.init(
    project="moveitpy_wandb",
)
wandb.log({"robot_state": wandb.Image(visualizer.get_robot_state_image(robot_state))})
wandb.log(
    {
        "robot_trajectory": wandb.Video(
            visualizer.get_robot_trajectory_images(plan.trajectory),
        ),
    },
)
wandb.finish()
