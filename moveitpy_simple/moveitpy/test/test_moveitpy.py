"""Tests for the MoveItPySimple class."""

from copy import deepcopy
from pathlib import Path

import numpy as np
from geometry_msgs.msg import Point, Pose, PoseStamped
from std_msgs.msg import Header

from moveitpy_simple.moveit_configs_utils import MoveItConfigsBuilder
from moveitpy_simple.moveitpy import (
    GripperState,
    MoveItPySimple,
    ValueRange,
    joint_positions_from_robot_state,
)

PANDA_GRIPPER_NUM_JOINTS = 1
PANDA_NUM_JOINTS = 7

path_dir = Path(__file__).parent


def test_moveitpy():
    """Test the MoveItPySimple class."""
    moveitpy = MoveItPySimple(
        "moveitpy_simple",
        MoveItConfigsBuilder(package=path_dir / "panda_moveit_config")
        .load_all()
        .to_moveit_configs(),
        arm_group_name="panda_arm",
        gripper_group_name="hand",
    )

    assert moveitpy.arm.joint_names == [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    ]
    assert moveitpy.gripper.joint_names == ["panda_finger_joint1"]
    assert moveitpy.joint_names == [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
        "panda_finger_joint1",
    ]
    assert len(moveitpy.gripper.get_joint_positions()) == PANDA_GRIPPER_NUM_JOINTS
    assert len(moveitpy.arm.get_joint_positions()) == PANDA_NUM_JOINTS
    assert (
        len(moveitpy.get_joint_positions())
        == PANDA_NUM_JOINTS + PANDA_GRIPPER_NUM_JOINTS
    )
    rs = moveitpy.robot_state()
    # TODO: Add a test with the robotiq gripper
    rs.joint_positions = {
        "panda_joint1": -1.5,
        "panda_joint2": -1.7628,
        "panda_joint3": 2.8,
        "panda_joint4": 0.0,
        "panda_joint5": 0.0,
        "panda_joint6": 3.7525,
        "panda_joint7": 0.5,
        "panda_finger_joint1": 0.035,
    }
    assert np.allclose(moveitpy.gripper.get_joint_positions(), [0.035])
    assert np.allclose(
        moveitpy.arm.get_joint_positions(),
        [-1.5, -1.7628, 2.8, 0.0, 0.0, 3.7525, 0.5],
    )
    assert np.allclose(
        moveitpy.get_joint_positions(),
        [-1.5, -1.7628, 2.8, 0.0, 0.0, 3.7525, 0.5, 0.035],
    )
    assert np.allclose(moveitpy.gripper.get_joint_positions(normalize=True), [1.0])
    assert np.allclose(
        moveitpy.arm.get_joint_positions(normalize=True),
        [-0.5177234, -1.0, 0.96642, 0.98867, 0.0, 1.0, 0.17257447],
    )
    rs.joint_positions = {
        "panda_finger_joint1": 0.0,
    }
    assert np.allclose(moveitpy.gripper.get_joint_positions(normalize=True), [-1.0])

    moveitpy.arm.set_goal_from_named_state("ready")
    assert (plan_result := moveitpy.arm.plan())

    moveitpy.arm.set_goal_from_pose_stamped(
        PoseStamped(
            pose=Pose(
                position=Point(x=0.2, y=-0.2, z=0.5),
            ),
            header=Header(frame_id="panda_link0"),
        ),
        "panda_link8",
    )
    assert (plan_result := moveitpy.arm.plan())

    rs = deepcopy(moveitpy.robot_state())
    rs.set_to_default_values("panda_arm", "extended")
    moveitpy.arm.set_goal_from_robot_state(rs)
    assert (plan_result := moveitpy.arm.plan())

    goal_joint_positions = [-1.5, -1.75, 2.0, 0.0, 0.0, 1.0, 0.75]
    moveitpy.arm.set_start_state("ready")
    moveitpy.arm.set_goal_from_joint_positions(goal_joint_positions)
    assert (plan_result := moveitpy.arm.plan())
    assert np.allclose(
        joint_positions_from_robot_state(
            plan_result.trajectory[len(plan_result.trajectory) - 1],
            moveitpy.arm.joint_names,
        ),
        goal_joint_positions,
        atol=1e-2,
    )

    goal_joint_positions = [-0.5, -1.0, 0.7, 1.0, 0.0, -0.45, 0.25]
    moveitpy.arm.set_start_state("ready")
    moveitpy.arm.set_goal_from_joint_positions(goal_joint_positions, normalize=True)
    assert (plan_result := moveitpy.arm.plan())
    assert np.allclose(
        moveitpy.arm.joint_positions_from_robot_state(
            plan_result.trajectory[len(plan_result.trajectory) - 1],
            normalize=True,
        ),
        goal_joint_positions,
        atol=1e-2,
    )

    moveitpy.arm.set_start_state("ready")
    moveitpy.arm.set_goal_from_joint_positions({"panda_joint1": -1.5})
    rs = deepcopy(moveitpy.arm.get_start_state())
    assert (plan_result := moveitpy.arm.plan())
    assert np.allclose(
        joint_positions_from_robot_state(rs, moveitpy.arm.joint_names),
        joint_positions_from_robot_state(
            plan_result.trajectory[0],
            moveitpy.arm.joint_names,
        ),
        atol=1e-2,
    )
    rs.joint_positions = {"panda_joint1": -1.5}
    assert np.allclose(
        joint_positions_from_robot_state(rs, moveitpy.arm.joint_names),
        joint_positions_from_robot_state(
            plan_result.trajectory[len(plan_result.trajectory) - 1],
            moveitpy.arm.joint_names,
        ),
        atol=1e-2,
    )


def test_auto_detect_groups():
    """Test that the groups are auto-detected from end_effectors srdf tag."""
    moveitpy = MoveItPySimple(
        "moveitpy_simple",
        MoveItConfigsBuilder(package=path_dir / "panda_moveit_config")
        .load_all()
        .to_moveit_configs(),
    )
    assert moveitpy.arm._joint_model_group.name == "panda_arm"
    assert moveitpy.gripper._joint_model_group.name == "hand"


def test_normalized_gripper_values():
    """Test the planning for the gripper jmg."""
    moveitpy = MoveItPySimple(
        "moveitpy_simple",
        MoveItConfigsBuilder(package=path_dir / "panda_moveit_config")
        .load_all()
        .to_moveit_configs(),
        gripper_value_range=ValueRange.NORMALIZED,
    )

    rs = moveitpy.robot_state()
    rs.joint_positions = {
        "panda_finger_joint1": 0.035,
    }
    assert np.allclose(moveitpy.gripper.get_joint_positions(), [0.035])
    assert np.allclose(moveitpy.gripper.get_joint_positions(normalize=True), [1.0])
    rs.joint_positions = {
        "panda_finger_joint1": 0.035 / 2.0,
    }
    assert np.allclose(moveitpy.gripper.get_joint_positions(normalize=True), [0.5])
    rs.joint_positions = {
        "panda_finger_joint1": 0.0,
    }
    assert np.allclose(moveitpy.gripper.get_joint_positions(), [0.0])
    assert np.allclose(moveitpy.gripper.get_joint_positions(normalize=True), [0.0])

    goal_joint_positions = [0.035]
    moveitpy.gripper.set_start_state(GripperState.CLOSE)
    # We don't have the current state monitor, so I manually set the start state to a valid state
    moveitpy.gripper.get_start_state().set_joint_group_positions(
        "panda_arm",
        [-1.5, -1.7628, 2.8, 0.0, 0.0, 3.7525, 0.5],
    )
    moveitpy.gripper.set_goal(goal_joint_positions)
    assert (plan_result := moveitpy.gripper.plan())
    assert np.allclose(
        joint_positions_from_robot_state(
            plan_result.trajectory[len(plan_result.trajectory) - 1],
            moveitpy.gripper.joint_names,
        ),
        goal_joint_positions,
        atol=1e-2,
    )
    assert np.allclose(
        moveitpy.gripper.joint_positions_from_robot_state(
            plan_result.trajectory[len(plan_result.trajectory) - 1],
            normalize=True,
        ),
        [1.0],
        atol=1e-2,
    )

    goal_joint_positions = [0.0]
    moveitpy.gripper.set_start_state(GripperState.OPEN)
    moveitpy.gripper.get_start_state().set_joint_group_positions(
        "panda_arm",
        [-1.5, -1.7628, 2.8, 0.0, 0.0, 3.7525, 0.5],
    )
    moveitpy.gripper.set_goal(goal_joint_positions, normalize=True)
    assert (plan_result := moveitpy.gripper.plan())
    assert np.allclose(
        moveitpy.gripper.joint_positions_from_robot_state(
            plan_result.trajectory[len(plan_result.trajectory) - 1],
        ),
        [0.0],
        atol=1e-2,
    )
    assert np.allclose(
        moveitpy.gripper.joint_positions_from_robot_state(
            plan_result.trajectory[len(plan_result.trajectory) - 1],
            normalize=True,
        ),
        goal_joint_positions,
        atol=1e-2,
    )

    goal_joint_positions = [0.25]
    moveitpy.gripper.set_start_state(GripperState.OPEN)
    moveitpy.gripper.get_start_state().set_joint_group_positions(
        "panda_arm",
        [-1.5, -1.7628, 2.8, 0.0, 0.0, 3.7525, 0.5],
    )
    moveitpy.gripper.set_goal(goal_joint_positions, normalize=True)
    assert (plan_result := moveitpy.gripper.plan())
    assert np.allclose(
        moveitpy.gripper.joint_positions_from_robot_state(
            plan_result.trajectory[len(plan_result.trajectory) - 1],
        ),
        [0.00875],
        atol=1e-2,
    )
    assert np.allclose(
        moveitpy.gripper.joint_positions_from_robot_state(
            plan_result.trajectory[len(plan_result.trajectory) - 1],
            normalize=True,
        ),
        goal_joint_positions,
        atol=1e-2,
    )
