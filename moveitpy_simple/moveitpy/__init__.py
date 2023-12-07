"""A simple wrapper around MoveItPy."""

from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum
from functools import partial
from typing import ClassVar

import numpy as np
from geometry_msgs.msg import Pose, PoseStamped
from moveit.core.kinematic_constraints import construct_joint_constraint
from moveit.core.planning_interface import MotionPlanResponse
from moveit.core.planning_scene import PlanningScene
from moveit.core.robot_model import JointModelGroup, RobotModel
from moveit.core.robot_state import RobotState
from moveit.core.robot_trajectory import RobotTrajectory
from moveit.core.time_optimal_trajectory_generation import (
    TimeOptimalTrajectoryGeneration,
)
from moveit.planning import MoveItPy, PlanningComponent, PlanningSceneMonitor
from moveit_msgs.msg import Constraints
from sensor_msgs.msg import JointState

from moveitpy_simple.moveit_configs_utils import MoveItConfigs


class ValueRange(Enum):
    """Value ranges."""

    UNIT: ClassVar[list[float]] = [-1.0, 1.0]
    NORMALIZED: ClassVar[list[float]] = [0.0, 1.0]


class GripperState(str, Enum):
    """Gripper states."""

    OPEN = "open"
    CLOSE = "close"


# TODO: Should we return a copy for the next two methods?
def get_planning_scene(planning_scene_monitor: PlanningSceneMonitor) -> PlanningScene:
    """Get the current planning scene of the planning_scene_monitor."""
    with planning_scene_monitor.read_only() as planning_scene:
        return planning_scene


def get_robot_state(planning_scene_monitor: PlanningSceneMonitor) -> RobotState:
    """Get the current robot state of the planning_scene_monitor."""
    return get_planning_scene(planning_scene_monitor).current_state


def create_joint_positions_converters(
    joint_names: list[str],
    joint_ranges: list[list[float]],
    value_range: ValueRange,
) -> tuple[dict, dict]:
    """Create joint positions normalizers and denormalizers."""
    joint_positions_normalizers = {}
    joint_positions_denormalizers = {}
    for joint_name, joint_positions_range in zip(
        joint_names,
        joint_ranges,
        strict=True,
    ):
        sorted_indices = np.argsort(joint_positions_range)
        joint_positions_normalizers[joint_name] = partial(
            np.interp,
            xp=np.take(joint_positions_range, sorted_indices),
            fp=np.take(value_range.value, sorted_indices),
        )
        joint_positions_denormalizers[joint_name] = partial(
            np.interp,
            xp=value_range.value,
            fp=joint_positions_range,
        )
    return joint_positions_normalizers, joint_positions_denormalizers


def joint_positions_from_robot_state(
    robot_state: RobotState,
    joint_names: list[str],
    normalizers: dict | None = None,
) -> list[float]:
    """Get joint positions from robot state."""
    if normalizers is not None:
        return [
            normalizers[joint_name](robot_state.joint_positions[joint_name])
            for joint_name in joint_names
        ]
    return [robot_state.joint_positions[joint_name] for joint_name in joint_names]


def filter_values_by_joint_names(
    keys: list[str],
    values: list[float],
    joint_names: list[str],
) -> list[float]:
    """Filter values by joint names."""
    filtered_values = []
    for joint_name in joint_names:
        try:
            index = keys.index(joint_name)
        except ValueError:
            msg = f"Joint name '{joint_name}' not in joint state msg."
            raise ValueError(msg) from None
        filtered_values.append(values[index])
    return filtered_values


class RobotComponent(ABC):
    """A class for a robot component."""

    def __init__(
        self,
        joint_model_group: JointModelGroup,
        planning_component: PlanningComponent,
        planning_scene_monitor: PlanningSceneMonitor,
        value_range: ValueRange,
    ) -> None:
        """Initialize a robot component."""
        self.joint_model_group = joint_model_group
        self._planning_component = planning_component
        self._planning_scene_monitor = planning_scene_monitor

        (
            # Convert from [min, max] to [-1, 1] or Convert [0, 1] to [min, max] and vice versa depending on the value range
            self._joint_positions_normalizers,
            # Convert from [-1, 1] to [min, max] or Convert [min, max] to [0, 1] depending on the value range
            self._joint_positions_denormalizers,
        ) = create_joint_positions_converters(
            self.joint_names,
            self.joint_limits,
            value_range,
        )

    @property
    @abstractmethod
    def joint_limits(self) -> list[list[float]]:
        """Joint limits for the joint model group."""

    @property
    def joint_names(self) -> list[str]:
        """Active joint names for the hand joint model group."""
        return self.joint_model_group.active_joint_model_names

    def get_joint_positions(self, *, normalize: bool = False) -> np.ndarray:
        """Get current joint positions for the gripper joint model group."""
        return joint_positions_from_robot_state(
            get_robot_state(self._planning_scene_monitor),
            self.joint_names,
            self._joint_positions_normalizers if normalize else None,
        )

    def joint_positions_from_robot_state(
        self,
        robot_state: RobotState,
        *,
        normalize: bool = False,
    ) -> np.ndarray:
        """Get joint positions from a robot state."""
        return joint_positions_from_robot_state(
            robot_state,
            self.joint_names,
            self._joint_positions_normalizers if normalize else None,
        )

    def get_named_joint_positions(self, name: str, *, normalize: bool = False) -> list:
        """Get named joint positions."""
        named_joint_positions = self._planning_component.get_named_target_state_values(
            name,
        )
        if normalize:
            return [
                self._joint_positions_normalizers[joint_name](
                    named_joint_positions[joint_name],
                )
                for joint_name in self.joint_names
            ]
        return [named_joint_positions[joint_name] for joint_name in self.joint_names]

    def joint_positions_from_joint_state_msg(
        self,
        joint_state_msg: JointState,
        *,
        normalize: bool = False,
    ) -> np.ndarray:
        """Get joint positions from a joint state msg."""
        positions = filter_values_by_joint_names(
            joint_state_msg.name,
            joint_state_msg.position,
            self.joint_names,
        )

        if normalize:
            return [
                self._joint_positions_normalizers[joint_name](position)
                for joint_name, position in zip(
                    self.joint_names,
                    positions,
                    strict=True,
                )
            ]
        return positions

    def joint_velocities_from_joint_state_msg(
        self,
        joint_state_msg: JointState,
    ) -> list[float]:
        """Get joint velocities from a joint state msg."""
        return filter_values_by_joint_names(
            joint_state_msg.name,
            joint_state_msg.velocity,
            self.joint_names,
        )

    def joint_efforts_from_joint_state_msg(
        self,
        joint_state_msg: JointState,
    ) -> list[float]:
        """Get joint efforts from a joint state msg."""
        return filter_values_by_joint_names(
            joint_state_msg.name,
            joint_state_msg.effort,
            self.joint_names,
        )

    def set_goal_from_named_state(self, named_state: str) -> None:
        """Set the goal to a named state."""
        self._planning_component.set_goal_state(configuration_name=named_state)

    def set_start_state(self, robot_state: RobotState | str) -> None:
        """Set the start state."""
        if isinstance(robot_state, str):
            self._planning_component.set_start_state(configuration_name=robot_state)
        elif isinstance(robot_state, RobotState):
            self._planning_component.set_start_state(robot_state=robot_state)
        else:
            msg = (
                f"robot_state must be a string or a RobotState, got {type(robot_state)}"
            )
            raise TypeError(
                msg,
            )

    def get_start_state(self) -> RobotState:
        """Get the start state."""
        return self._planning_component.get_start_state()

    def plan(self) -> MotionPlanResponse:
        """Plan a trajectory to the goal."""
        return self._planning_component.plan()

    def normalize_joint_positions(self, joint_positions: list[float]) -> list[float]:
        """Normalize joint positions."""
        return np.asarray(
            [
                self._joint_positions_normalizers[joint_name](position)
                for joint_name, position in zip(
                    self.joint_names,
                    joint_positions,
                    strict=True,
                )
            ],
        )

    def denormalize_joint_positions(self, joint_positions: list[float]) -> list[float]:
        """Denormalize joint positions."""
        return np.asarray(
            [
                self._joint_positions_denormalizers[joint_name](position)
                for joint_name, position in zip(
                    self.joint_names,
                    joint_positions,
                    strict=True,
                )
            ],
        )


class Gripper(RobotComponent):
    """A class that wraps around a gripper joint model group & planning component."""

    def __init__(
        self,
        joint_model_group: JointModelGroup,
        planning_component: PlanningComponent,
        planning_scene_monitor: PlanningSceneMonitor,
        value_range: ValueRange | None = None,
    ) -> None:
        """Initialize the gripper.

        Args:
            joint_model_group: The joint model group of the gripper.
            planning_component: The planning component of the gripper.
            planning_scene_monitor: The planning scene monitor.
            value_range: The value range of the gripper joint positions, defaults to ValueRange.UNIT.
        """
        super().__init__(
            joint_model_group,
            planning_component,
            planning_scene_monitor,
            value_range or ValueRange.UNIT,
        )

    @property
    def joint_limits(self) -> list[list[float]]:
        """Joint limits for the joint model group."""
        for gripper_state in GripperState:
            if gripper_state.value not in self._planning_component.named_target_states:
                msg = f"Gripper joint model group does not have named target {gripper_state.value}"
                raise ValueError(
                    msg,
                )
        gripper_open_joint_positions = (
            self._planning_component.get_named_target_state_values(
                GripperState.OPEN,
            )
        )
        gripper_close_joint_positions = (
            self._planning_component.get_named_target_state_values(
                GripperState.CLOSE,
            )
        )
        return [
            [
                gripper_close_joint_positions[joint_name],
                gripper_open_joint_positions[joint_name],
            ]
            for joint_name in self.joint_names
        ]

    def set_goal(
        self,
        joint_positions: list[float],
        *,
        normalized: bool = False,
    ) -> None:
        """Set the goal to a joint positions.

        Args:
            joint_positions: The joint positions.
            normalized: Whether the joint positions are normalized, defaults to False.
        """
        goal_joint_positions = {
            joint_name: (
                self._joint_positions_denormalizers[joint_name](joint_position)
                if normalized
                else joint_position
            )
            for joint_name, joint_position in zip(
                self.joint_names,
                joint_positions,
                strict=True,
            )
        }
        robot_state = deepcopy(self._planning_component.get_start_state())
        robot_state.joint_positions = goal_joint_positions
        joint_constraint = construct_joint_constraint(
            robot_state=robot_state,
            joint_model_group=self.joint_model_group,
            # TODO: Why MoveItPy uses 0.01 as tolerance? MoveIt set it to std::numeric_limits<double>::epsilon() by default
            tolerance=np.finfo(np.float32).eps,
        )
        self._planning_component.set_goal_state(
            motion_plan_constraints=[joint_constraint],
        )


class Arm(RobotComponent):
    """A class that wraps around an arm joint model group & planning component."""

    def __init__(
        self,
        robot_model: RobotModel,
        joint_model_group: JointModelGroup,
        planning_component: PlanningComponent,
        planning_scene_monitor: PlanningSceneMonitor,
    ) -> None:
        """Initialize the arm.

        Args:
            robot_model: The robot model.
            joint_model_group: The joint model group of the arm.
            planning_component: The planning component of the arm.
            planning_scene_monitor: The planning scene monitor.
        """
        self._robot_model = robot_model
        super().__init__(
            joint_model_group=joint_model_group,
            planning_component=planning_component,
            planning_scene_monitor=planning_scene_monitor,
            value_range=ValueRange.UNIT,
        )

    @property
    def joint_limits(self) -> list[list[float]]:
        """Joint limits for the joint model group."""
        return [
            [
                joint_limit[0].min_position,
                joint_limit[0].max_position,
            ]
            for joint_limit in self.joint_model_group.active_joint_model_bounds
        ]

    @property
    def velocity_limits(self) -> list[float]:
        """Velocity limits for the joint model group."""
        return [
            [
                joint_limit[0].min_velocity,
                joint_limit[0].max_velocity,
            ]
            for joint_limit in self.joint_model_group.active_joint_model_bounds
        ]

    # TODO: We should have a way to specify (Already possible with multi_plan_parameters/single_plan_parameters)
    # - Planning time
    # - Number of planning attempts
    # - Max velocity scaling factor
    # - Max acceleration scaling factor
    # TODO: computeCartesianPath -- Need a pybind11 support first
    def set_goal_from_robot_state(self, robot_state: RobotState) -> None:
        """Set the goal to a robot state."""
        self._planning_component.set_goal_state(robot_state=robot_state)

    def set_goal_from_pose_stamped(
        self,
        pose_stamped: PoseStamped,
        link_name: str,
    ) -> None:
        """Set the goal to a pose stamped."""
        self._planning_component.set_goal_state(
            pose_stamped_msg=pose_stamped,
            pose_link=link_name,
        )

    def set_goal_from_joint_positions(
        self,
        joint_positions: dict[str, float] | list[float],
        *,
        normalized: bool = False,
    ) -> None:
        """Set the goal to a joint positions.

        Args:
            joint_positions: The joint positions.
            normalized: Whether the joint positions are normalized (in [0 1] or [-1 1]), defaults to False.
        """
        goal_joint_positions = {}
        if isinstance(joint_positions, dict):
            if normalized:
                for joint_name, joint_position in joint_positions.items():
                    goal_joint_positions[
                        joint_name
                    ] = self._joint_positions_denormalizers[joint_name](joint_position)
            else:
                goal_joint_positions = joint_positions
        if isinstance(joint_positions, list | np.ndarray):
            for joint_name, joint_position in zip(
                self.joint_names,
                joint_positions,
                strict=True,
            ):
                goal_joint_positions[joint_name] = (
                    self._joint_positions_denormalizers[joint_name](joint_position)
                    if normalized
                    else joint_position
                )
        robot_state = deepcopy(self._planning_component.get_start_state())
        robot_state.joint_positions = goal_joint_positions
        joint_constraint = construct_joint_constraint(
            robot_state=robot_state,
            joint_model_group=self.joint_model_group,
            # TODO: Why MoveItPy uses 0.01 as tolerance? MoveIt set it to std::numeric_limits<double>::epsilon() by default
            tolerance=np.finfo(np.float32).eps,
        )
        self._planning_component.set_goal_state(
            motion_plan_constraints=[joint_constraint],
        )

    def set_goal_from_constraints(self, constraints: list[Constraints]) -> None:
        """Set the goal to a set of constraints."""
        self._planning_component.set_goal_state(motion_plan_constraints=constraints)

    def ik(self, pose: list[float], link_name: str) -> list[float] | None:
        """Compute the inverse kinematics of a pose.

        Args:
            pose: The pose [x, y, z, qx, qy, qz, qw].
            link_name: The link name.

        Returns:
            The joint positions.
        """
        robot_state = RobotState(self._robot_model)
        robot_state.set_to_default_values()
        pose_msg = Pose()
        pose_msg.position.x = float(pose[0])
        pose_msg.position.y = float(pose[1])
        pose_msg.position.z = float(pose[2])
        pose_msg.orientation.x = float(pose[3])
        pose_msg.orientation.y = float(pose[4])
        pose_msg.orientation.z = float(pose[5])
        pose_msg.orientation.w = float(pose[6])
        if robot_state.set_from_ik(
            self._planning_component.planning_group_name,
            pose_msg,
            link_name,
        ):
            return self.joint_positions_from_robot_state(robot_state)
        return None


class MoveItPySimple:
    """A class to simplify the usage of MoveItPy."""

    def __init__(  # noqa: PLR0913
        self,
        node_name: str,
        moveit_configs: MoveItConfigs,
        *,
        arm_group_name: str | None = None,
        gripper_group_name: str | None = None,
        gripper_value_range: ValueRange | None = None,
    ) -> None:
        """Initialize the MoveItPySimple class."""
        self._moveit_py = MoveItPy(node_name, config_dict=moveit_configs.to_dict())
        self.robot_model: RobotModel = self._moveit_py.get_robot_model()

        if arm_group_name is None and gripper_group_name is None:
            from srdfdom.srdf import SRDF

            srdf = SRDF.from_xml_string(
                moveit_configs.robot_description_semantic["robot_description_semantic"],
            )
            end_effectors = srdf.end_effectors
            if len(end_effectors) != 1:
                msg = (
                    "Can't infer the arm and gripper group name from the SRDF, please specify arm_group_name and gripper_group_name"
                    "Available end effectors: "
                    f"{[(end_effector.name, end_effector.group, end_effector.parent_group) for end_effector in end_effectors]}"
                )
                raise ValueError(
                    msg,
                )
            gripper_group_name = end_effectors[0].group
            arm_group_name = end_effectors[0].parent_group
        if not self.robot_model.has_joint_model_group(arm_group_name):
            msg = f"Robot model does not have group {arm_group_name} defined"
            raise ValueError(
                msg,
            )
        if not self.robot_model.has_joint_model_group(gripper_group_name):
            msg = f"Robot model does not have group {gripper_group_name} defined"
            raise ValueError(
                msg,
            )

        self.gripper = Gripper(
            self.robot_model.get_joint_model_group(gripper_group_name),
            self._moveit_py.get_planning_component(gripper_group_name),
            self._moveit_py.get_planning_scene_monitor(),
            value_range=gripper_value_range,
        )

        self.arm = Arm(
            self.robot_model,
            self.robot_model.get_joint_model_group(arm_group_name),
            self._moveit_py.get_planning_component(arm_group_name),
            self._moveit_py.get_planning_scene_monitor(),
        )

    @property
    def joint_names(self) -> list[str]:
        """Active joint names for the robot."""
        return self.arm.joint_names + self.gripper.joint_names

    def planning_scene(self) -> PlanningScene:
        """Get a reference to the current planning scene."""
        return get_planning_scene(self._moveit_py.get_planning_scene_monitor())

    def robot_state(self) -> RobotState:
        """Get the reference to the current robot state."""
        return get_robot_state(self._moveit_py.get_planning_scene_monitor())

    def get_joint_positions(self, *, normalize: bool = False) -> np.ndarray:
        """Get current joint positions for the arm and gripper joint model groups."""
        return np.concatenate(
            (
                self.arm.get_joint_positions(normalize=normalize),
                self.gripper.get_joint_positions(normalize=normalize),
            ),
        )

    def execute(self, trajectory: RobotTrajectory, *, blocking: bool = True) -> None:
        """Execute a trajectory."""
        return self._moveit_py.execute(trajectory, blocking=blocking)

    def get_pose(
        self,
        link_name: str,
        robot_state: list | RobotState | None = None,
    ) -> np.ndarray:
        """Get the pose of a link."""
        if robot_state is None:
            robot_state = self.robot_state()
        if isinstance(robot_state, RobotState):
            robot_state.update()
            return robot_state.get_global_link_transform(link_name)
        elif isinstance(robot_state, list | np.ndarray):  # noqa: RET505
            assert len(robot_state) == len(
                self.joint_names,
            ), f"Wrong number of joint positions: {robot_state} != {self.joint_names}"
            assert len(robot_state[: len(self.arm.joint_names)]) == len(
                self.arm.joint_names,
            ), f"Wrong number of joint positions for arm: {robot_state[: len(self.arm.joint_names)]} != {self.arm.joint_names}"
            assert len(robot_state[len(self.arm.joint_names) :]) == len(
                self.gripper.joint_names,
            ), f"Wrong number of joint positions for gripper: {robot_state[len(self.arm.joint_names) :]} != {self.gripper.joint_names}"
            rs = RobotState(self.robot_model)
            rs.set_to_default_values()
            rs.set_joint_group_active_positions(
                self.arm.joint_model_group.name,
                robot_state[: len(self.arm.joint_names)],
            )
            rs.set_joint_group_active_positions(
                self.gripper.joint_model_group.name,
                robot_state[len(self.arm.joint_names) :],
            )
            rs.update()
            return rs.get_global_link_transform(link_name)
        else:
            msg = f"robot_state must be either a RobotState or a list of joint positions -- got {type(robot_state)}"
            raise TypeError(
                msg,
            )

    def joint_positions_from_robot_state(
        self,
        robot_state: RobotState,
        *,
        normalize: bool = False,
    ) -> list[float]:
        """Get the joint positions from a RobotState object."""
        return np.concatenate(
            [
                self.arm.joint_positions_from_robot_state(
                    robot_state,
                    normalize=normalize,
                ),
                self.gripper.joint_positions_from_robot_state(
                    robot_state,
                    normalize=normalize,
                ),
            ],
        )

    def joint_positions_from_joint_state_msg(
        self,
        joint_state_msg: JointState,
        *,
        normalize: bool = False,
    ) -> list[float]:
        """Get the joint positions from a JointState message."""
        return np.concatenate(
            [
                self.arm.joint_positions_from_joint_state_msg(
                    joint_state_msg,
                    normalize=normalize,
                ),
                self.gripper.joint_positions_from_joint_state_msg(
                    joint_state_msg,
                    normalize=normalize,
                ),
            ],
        )

    def joint_velocities_from_joint_state_msg(
        self,
        joint_state_msg: JointState,
    ) -> np.ndarray:
        """Get joint velocities from a joint state msg."""
        return np.concatenate(
            [
                self.arm.joint_velocities_from_joint_state_msg(joint_state_msg),
                self.gripper.joint_velocities_from_joint_state_msg(joint_state_msg),
            ],
        )

    def joint_efforts_from_joint_state_msg(
        self,
        joint_state_msg: JointState,
    ) -> np.ndarray:
        """Get joint efforts from a joint state msg."""
        return np.concatenate(
            [
                self.arm.joint_efforts_from_joint_state_msg(joint_state_msg),
                self.gripper.joint_efforts_from_joint_state_msg(joint_state_msg),
            ],
        )

    def make_robot_state(self, joint_positions: list[float]) -> RobotState:
        """Create a robot state from joint positions."""
        assert len(joint_positions) == len(
            self.arm.joint_names,
        ), f"Wrong number of joint positions: {joint_positions} != {self.arm.joint_names}"
        robot_state = RobotState(self.robot_model)
        robot_state.set_to_default_values()
        robot_state.set_joint_group_positions(
            self.arm.joint_model_group.name,
            joint_positions[: len(self.arm.joint_names)],
        )
        # TODO: Support setting gripper joint positions
        return robot_state

    def make_robot_trajectory(
        self,
        joint_trajectory: list[list[float]] | list[RobotState] | list[np.ndarray],
        resample_dt: float = 0.1,
    ) -> RobotTrajectory:
        """Create a robot trajectory from joint positions."""
        assert len(joint_trajectory) > 0, "Empty trajectory"
        robot_trajectory = RobotTrajectory(
            self.robot_model,
            self.arm.joint_model_group.name,
        )
        if isinstance(joint_trajectory[0], RobotState):
            for robot_state in joint_trajectory:
                robot_trajectory.add_suffix_waypoint(robot_state, 0.0)
        elif isinstance(joint_trajectory[0], list | np.ndarray):
            for joint_positions in joint_trajectory:
                robot_trajectory.add_suffix_waypoint(
                    self.make_robot_state(joint_positions),
                    0.0,
                )
        else:
            msg = f"joint_trajectory must be a list of RobotState or a list of list of joint positions -- got {type(joint_trajectory)}"
            raise TypeError(
                msg,
            )
        totg = TimeOptimalTrajectoryGeneration(resample_dt=resample_dt)
        if not totg.compute_time_stamps(robot_trajectory):
            return None
        return robot_trajectory

    def is_state_valid(
        self,
        robot_state: RobotState | list[float] | np.ndarray,
    ) -> bool:
        """Check if a robot state is valid."""
        # We need to make a copy of the planning scene since we will modify it
        # which avoid changing the main planning scene in the planning scene monitor
        planning_scene = deepcopy(self.planning_scene())
        if isinstance(robot_state, RobotState):
            rs = robot_state
        elif isinstance(robot_state, list | np.ndarray):
            assert len(robot_state) == len(
                self.joint_names,
            ), f"Wrong number of joint positions: {robot_state} != {self.joint_names}"
            assert len(robot_state[: len(self.arm.joint_names)]) == len(
                self.arm.joint_names,
            ), f"Wrong number of joint positions for arm: {robot_state[: len(self.arm.joint_names)]} != {self.arm.joint_names}"
            assert len(robot_state[len(self.arm.joint_names) :]) == len(
                self.gripper.joint_names,
            ), f"Wrong number of joint positions for gripper: {robot_state[len(self.arm.joint_names) :]} != {self.gripper.joint_names}"
            rs = RobotState(self.robot_model)
            rs.set_to_default_values()
            rs.set_joint_group_active_positions(
                self.arm.joint_model_group.name,
                robot_state[: len(self.arm.joint_names)],
            )
            rs.set_joint_group_active_positions(
                self.gripper.joint_model_group.name,
                robot_state[len(self.arm.joint_names) :],
            )
        else:
            msg = f"robot_state must be either a RobotState or a list of joint positions -- got {type(robot_state)}"
            raise TypeError(
                msg,
            )
        rs.update()
        return (
            planning_scene.is_state_valid(
                rs,
                self.arm.joint_model_group.name,
            )
            and planning_scene.is_state_valid(
                rs,
                self.gripper.joint_model_group.name,
            )
            and self.arm.joint_model_group.satisfies_position_bounds(
                self.arm.joint_positions_from_robot_state(rs),
            )
            and self.gripper.joint_model_group.satisfies_position_bounds(
                self.gripper.joint_positions_from_robot_state(rs),
            )
        )
