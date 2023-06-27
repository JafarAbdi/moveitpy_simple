"""A simple wrapper around MoveItPy."""

from copy import deepcopy
from enum import Enum
from functools import partial
from typing import ClassVar

import numpy as np
from geometry_msgs.msg import PoseStamped
from moveit.core.kinematic_constraints import construct_joint_constraint
from moveit.core.planning_interface import MotionPlanResponse
from moveit.core.planning_scene import PlanningScene
from moveit.core.robot_model import JointModelGroup, RobotModel
from moveit.core.robot_state import RobotState
from moveit.core.robot_trajectory import RobotTrajectory
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
        joint_positions_normalizers[joint_name] = partial(
            np.interp,
            xp=joint_positions_range,
            fp=value_range.value,
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


def joint_positions_from_joint_state_msg(
    joint_state_msg: JointState, joint_names: list[str], normalizers: dict | None = None
) -> list[float]:
    """Get joint positions from joint state msg."""
    positions = [
        joint_state_msg.position[joint_state_msg.name.index(joint_name)]
        for joint_name in joint_names
    ]
    if normalizers is not None:
        return [
            normalizers[joint_name](position)
            for joint_name, position in zip(joint_names, positions)
        ]
    return positions


class Gripper:
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
        for gripper_state in GripperState:
            if gripper_state.value not in planning_component.named_target_states:
                msg = f"Gripper joint model group does not have named target {gripper_state.value}"
                raise ValueError(
                    msg,
                )

        self.joint_model_group = joint_model_group
        self._planning_component = planning_component
        self._planning_scene_monitor = planning_scene_monitor
        self._value_range = value_range or ValueRange.UNIT

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
        gripper_values_range = [
            [
                gripper_close_joint_positions[joint_name],
                gripper_open_joint_positions[joint_name],
            ]
            for joint_name in self.joint_names
        ]

        (
            # Convert from [min, max] to [-1, 1] or Convert [0, 1] to [min, max] and vice versa depending on the value range
            self._joint_positions_normalizers,
            # Convert from [-1, 1] to [min, max] or Convert [min, max] to [0, 1] depending on the value range
            self._joint_positions_denormalizers,
        ) = create_joint_positions_converters(
            self.joint_names,
            gripper_values_range,
            self._value_range,
        )

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

    def joint_positions_from_joint_state_msg(
        self,
        joint_state_msg: JointState,
        *,
        normalize: bool = False,
    ) -> np.ndarray:
        """Get joint positions from a joint state msg."""
        return joint_positions_from_joint_state_msg(
            joint_state_msg,
            self.joint_names,
            self._joint_positions_normalizers if normalize else None,
        )

    def set_start_state(self, robot_state: RobotState | str) -> None:
        """Set the start state."""
        self._planning_component.set_start_state(configuration_name=robot_state)

    def get_start_state(self) -> RobotState:
        """Get the start state."""
        return self._planning_component.get_start_state()

    def set_goal(
        self,
        joint_positions: list[float],
        *,
        normalize: bool = False,
    ) -> None:
        """Set the goal to a joint positions."""
        goal_joint_positions = {
            joint_name: (
                self._joint_positions_denormalizers[joint_name](joint_position)
                if normalize
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

    def plan(self) -> MotionPlanResponse:
        """Plan a trajectory to the goal."""
        return self._planning_component.plan()


class Arm:
    """A class that wraps around an arm joint model group & planning component."""

    def __init__(
        self,
        joint_model_group: JointModelGroup,
        planning_component: PlanningComponent,
        planning_scene_monitor: PlanningSceneMonitor,
    ) -> None:
        """Initialize the arm.

        Args:
            joint_model_group: The joint model group of the arm.
            planning_component: The planning component of the arm.
            planning_scene_monitor: The planning scene monitor.
        """
        self.joint_model_group = joint_model_group
        self._planning_component = planning_component
        self._planning_scene_monitor = planning_scene_monitor

        (
            # Convert from [min, max] to [-1, 1]
            self._joint_positions_normalizers,
            # Convert from [-1, 1] to [min, max]
            self._joint_positions_denormalizers,
        ) = create_joint_positions_converters(
            self.joint_names,
            [
                [
                    joint_limit[0].min_position,
                    joint_limit[0].max_position,
                ]
                for joint_limit in self.joint_model_group.active_joint_model_bounds
            ],
            ValueRange.UNIT,
        )

    @property
    def joint_names(self) -> list[str]:
        """Active joint names for the arm joint model group."""
        return self.joint_model_group.active_joint_model_names

    def get_joint_positions(self, *, normalize: bool = False) -> np.ndarray:
        """Get current joint positions for the arm joint model group."""
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

    def joint_positions_from_joint_state_msg(
        self,
        joint_state_msg: JointState,
        *,
        normalize: bool = False,
    ) -> np.ndarray:
        """Get joint positions from a joint state message."""
        return joint_positions_from_joint_state_msg(
            joint_state_msg,
            self.joint_names,
            self._joint_positions_normalizers if normalize else None,
        )

    # TODO: We should have a way to specify (Already possible with multi_plan_parameters/single_plan_parameters)
    # - Planning time
    # - Number of planning attempts
    # - Max velocity scaling factor
    # - Max acceleration scaling factor
    # TODO: computeCartesianPath -- Need a pybind11 support first
    def set_goal_from_named_state(self, named_state: str) -> None:
        """Set the goal to a named state."""
        self._planning_component.set_goal_state(configuration_name=named_state)

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
        normalize: bool = False,
    ) -> None:
        """Set the goal to a joint positions."""
        goal_joint_positions = {}
        if isinstance(joint_positions, dict):
            if normalize:
                for joint_name, joint_position in joint_positions.items():
                    goal_joint_positions[
                        joint_name
                    ] = self._joint_positions_denormalizers[joint_name](joint_position)
            else:
                goal_joint_positions = joint_positions
        if isinstance(joint_positions, list):
            for joint_name, joint_position in zip(
                self.joint_names,
                joint_positions,
                strict=True,
            ):
                goal_joint_positions[joint_name] = (
                    self._joint_positions_denormalizers[joint_name](joint_position)
                    if normalize
                    else joint_position
                )
        robot_state = deepcopy(self._planning_component.get_start_state())
        robot_state.joint_positions = goal_joint_positions
        joint_constraint = construct_joint_constraint(
            robot_state=robot_state,
            joint_model_group=self.joint_model_group,
        )
        self._planning_component.set_goal_state(
            motion_plan_constraints=[joint_constraint],
        )

    def set_goal_from_constraints(self, constraints: list[Constraints]) -> None:
        """Set the goal to a set of constraints."""
        self._planning_component.set_goal_state(motion_plan_constraints=constraints)

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
                msg = "Can't infer the arm and gripper group name from the SRDF, please specify arm_group_name and gripper_group_name"
                raise ValueError(
                    msg,
                )
            gripper_group_name = end_effectors[0].name
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

    def execute(self, trajectory: RobotTrajectory) -> None:
        """Execute a trajectory."""
        self._moveit_py.execute(trajectory, controllers=[])
