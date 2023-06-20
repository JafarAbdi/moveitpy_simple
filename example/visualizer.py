#!/usr/bin/env python3

"""A simple example of using moveitpy with panda3d_viewer and wandb."""

from pathlib import Path

from moveit.core.robot_state import RobotState

import wandb
from moveitpy_simple.moveit_configs_utils import MoveItConfigsBuilder
from moveitpy_simple.moveitpy import MoveItPySimple
from moveitpy_simple.moveitpy.visualization import Visualizer

dir_path = Path(__file__).parent.absolute()


# TODO: This won't work for an installed package
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

moveitpy = MoveItPySimple(
    "moveitpy_simple",
    moveit_configs,
)


robot_state = RobotState(moveitpy.robot_model)
robot_state.set_to_default_values(moveitpy.arm.joint_model_group, "ready")
robot_state.set_to_default_values(moveitpy.gripper.joint_model_group, "close")
robot_state.update()

visualizer = Visualizer(moveit_configs.robot_description["robot_description"])
visualizer.visualize_robot_state(robot_state)

moveitpy.arm.set_start_state(robot_state)
moveitpy.arm.set_goal_from_named_state("extended")
plan = moveitpy.arm.plan()
if plan:
    visualizer.visualize_robot_trajectory(plan.trajectory)


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
