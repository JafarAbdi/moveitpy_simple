from copy import deepcopy
from enum import Enum
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
from geometry_msgs.msg import PoseStamped
from moveit.core.kinematic_constraints import construct_joint_constraint
from moveit.core.planning_interface import MotionPlanResponse
from moveit.core.planning_scene import PlanningScene
from moveit.core.robot_state import RobotState
from moveit.core.robot_trajectory import RobotTrajectory
from moveit.planning import MoveItPy, PlanningComponent
from moveit_msgs.msg import Constraints

from moveitpy_simple.moveit_configs_utils import MoveItConfigs

if TYPE_CHECKING:
    from moveit.core.robot_model import JointModelGroup, RobotModel


class GripperState(str, Enum):
    OPEN = "open"
    CLOSE = "close"


# TODO: I think we should have a class to wrap the arm and gripper planning components
# gripper.open() Do we really need these? The move should be enough
# gripper.move_async()?


# TODO: Make it so we could have MoveItPy without trajectory execution manager
# TODO: Make it possible to load directly from a package name
# TODO: Make gripper optional?
class MoveItPySimple:
    def __init__(
        self,
        node_name: str,
        moveit_configs: MoveItConfigs,
        arm_group_name: str,
        gripper_group_name: str,
    ) -> None:
        self._moveit_py = MoveItPy(node_name, config_dict=moveit_configs.to_dict())
        self._robot_model: RobotModel = self._moveit_py.get_robot_model()
        if not self._robot_model.has_joint_model_group(arm_group_name):
            msg = f"Robot model does not have group {arm_group_name} defined"
            raise ValueError(
                msg,
            )
        if not self._robot_model.has_joint_model_group(gripper_group_name):
            msg = f"Robot model does not have group {gripper_group_name} defined"
            raise ValueError(
                msg,
            )
        self._arm_joint_model_group: JointModelGroup = (
            self._robot_model.get_joint_model_group(arm_group_name)
        )
        self._gripper_joint_model_group: JointModelGroup = (
            self._robot_model.get_joint_model_group(gripper_group_name)
        )
        self._arm_planning_component: PlanningComponent = (
            self._moveit_py.get_planning_component(arm_group_name)
        )
        self._gripper_planning_component: PlanningComponent = (
            self._moveit_py.get_planning_component(gripper_group_name)
        )

        for gripper_state in GripperState:
            if (
                gripper_state.value
                not in self._gripper_planning_component.named_target_states
            ):
                msg = f"Gripper joint model group does not have named target {gripper_state.value}"
                raise ValueError(
                    msg,
                )

        fraction_range = [0, 1]
        normalized_joint_positions_range = [-1, 1]
        # Convert from [min, max] to [-1, 1]
        self._joint_positions_normalizers = {}
        # Convert from [-1, 1] to [min, max]
        self._joint_positions_denormalizers = {}
        for joint_name, joint_limit in zip(
            self.arm_joint_names,
            self._arm_joint_model_group.active_joint_model_bounds,
            strict=True,
        ):
            joint_positions_range = [
                joint_limit[0].min_position,
                joint_limit[0].max_position,
            ]
            self._joint_positions_normalizers[joint_name] = partial(
                np.interp,
                xp=joint_positions_range,
                fp=normalized_joint_positions_range,
            )
            self._joint_positions_denormalizers[joint_name] = partial(
                np.interp,
                xp=normalized_joint_positions_range,
                fp=joint_positions_range,
            )

        # Convert [0, 1] to [min, max]
        self._gripper_fraction_to_joint_position = {}
        self._gripper_joint_position_to_fraction = {}
        gripper_open_joint_positions = (
            self._gripper_planning_component.get_named_target_state_values(
                GripperState.OPEN,
            )
        )
        gripper_close_joint_positions = (
            self._gripper_planning_component.get_named_target_state_values(
                GripperState.CLOSE,
            )
        )
        for joint_name in self.gripper_joint_names:
            joint_positions_range = [
                gripper_close_joint_positions[joint_name],
                gripper_open_joint_positions[joint_name],
            ]
            self._gripper_fraction_to_joint_position[joint_name] = partial(
                np.interp,
                xp=fraction_range,
                fp=joint_positions_range,
            )
            self._gripper_joint_position_to_fraction[joint_name] = partial(
                np.interp,
                xp=joint_positions_range,
                fp=fraction_range,
            )
            self._joint_positions_normalizers[joint_name] = partial(
                np.interp,
                xp=joint_positions_range,
                fp=normalized_joint_positions_range,
            )
            self._joint_positions_denormalizers[joint_name] = partial(
                np.interp,
                xp=normalized_joint_positions_range,
                fp=joint_positions_range,
            )

    @property
    def arm_joint_names(self):
        """Active joint names for the arm joint model group."""
        return self._arm_joint_model_group.active_joint_model_names

    @property
    def gripper_joint_names(self):
        """Active joint names for the hand joint model group."""
        return self._gripper_joint_model_group.active_joint_model_names

    @property
    def joint_names(self):
        """Active joint names for the robot."""
        return self.arm_joint_names + self.gripper_joint_names

    # TODO: Should we return a copy for the next two methods?
    def planning_scene(self) -> PlanningScene:
        """Get a reference to the current planning scene."""
        with self._moveit_py.get_planning_scene_monitor().read_only() as planning_scene:
            return planning_scene

    def robot_state(self) -> RobotState:
        """Get the reference to the current robot state."""
        return self.planning_scene().current_state

    def joint_positions_from_robot_state(
        self,
        robot_state: RobotState,
        joint_names: list[str],
        normalized: bool = False,
    ) -> np.ndarray:
        joint_positions = []
        for joint_name in joint_names:
            joint_position = robot_state.joint_positions[joint_name]
            joint_positions.append(
                self._joint_positions_normalizers[joint_name](joint_position)
                if normalized
                else joint_position,
            )
        return np.asarray(joint_positions)

    def get_arm_joint_positions(self, normalized: bool = False) -> np.ndarray:
        """Get current joint positions for the arm joint model group."""
        return self.joint_positions_from_robot_state(
            self.robot_state(),
            self.arm_joint_names,
            normalized,
        )

    def get_gripper_joint_positions(self, normalized: bool = False) -> np.ndarray:
        """Get current joint positions for the gripper joint model group."""
        return self.joint_positions_from_robot_state(
            self.robot_state(),
            self.gripper_joint_names,
            normalized,
        )

    def get_gripper_state(self) -> np.ndarray:
        """Get current gripper state."""
        gripper_joint_positions = self.get_gripper_joint_positions()
        gripper_state = [
            self._gripper_joint_position_to_fraction[joint_name](
                gripper_joint_position,
            )
            for joint_name, gripper_joint_position in zip(
                self.gripper_joint_names,
                gripper_joint_positions,
                strict=True,
            )
        ]
        return np.asarray(gripper_state)

    def get_joint_positions(self, normalized=False) -> np.ndarray:
        """Get current joint positions for the arm and gripper joint model groups."""
        return np.concatenate(
            (
                self.get_arm_joint_positions(normalized),
                self.get_gripper_joint_positions(normalized),
            ),
        )

    # TODO: We should have a way to specify (Already possible with multi_plan_parameters/single_plan_parameters)
    # - Planning time
    # - Number of planning attempts
    # - Max velocity scaling factor
    # - Max acceleration scaling factor
    # TODO: computeCartesianPath -- Need a pybind11 support first
    def set_goal_from_named_state(self, named_state: str):
        """Set the goal to a named state."""
        self._arm_planning_component.set_goal_state(configuration_name=named_state)

    def set_goal_from_robot_state(self, robot_state: RobotState):
        """Set the goal to a robot state."""
        self._arm_planning_component.set_goal_state(robot_state=robot_state)

    def set_goal_from_pose_stamped(self, pose_stamped: PoseStamped, link_name: str):
        """Set the goal to a pose stamped."""
        self._arm_planning_component.set_goal_state(
            pose_stamped_msg=pose_stamped,
            pose_link=link_name,
        )

    def set_goal_from_joint_positions(
        self,
        joint_positions: dict[str, float] | list[float],
        normalized: bool = False,
    ):
        """Set the goal to a joint positions."""
        goal_joint_positions = {}
        if isinstance(joint_positions, dict):
            if normalized:
                for joint_name, joint_position in joint_positions.items():
                    goal_joint_positions[
                        joint_name
                    ] = self._joint_positions_denormalizers[joint_name](joint_position)
            else:
                goal_joint_positions = joint_positions
        if isinstance(joint_positions, list):
            for joint_name, joint_position in zip(
                self.arm_joint_names,
                joint_positions,
                strict=True,
            ):
                goal_joint_positions[joint_name] = (
                    self._joint_positions_denormalizers[joint_name](joint_position)
                    if normalized
                    else joint_position
                )
        robot_state = deepcopy(self._arm_planning_component.get_start_state())
        robot_state.joint_positions = goal_joint_positions
        joint_constraint = construct_joint_constraint(
            robot_state=robot_state,
            joint_model_group=self._arm_joint_model_group,
        )
        self._arm_planning_component.set_goal_state(
            motion_plan_constraints=[joint_constraint],
        )

    def set_goal_from_constraints(self, constraints: list[Constraints]):
        """Set the goal to a set of constraints."""
        self._arm_planning_component.set_goal_state(motion_plan_constraints=constraints)

    def set_start_state(self, robot_state: RobotState | str):
        """Set the start state."""
        if isinstance(robot_state, str):
            self._arm_planning_component.set_start_state(configuration_name=robot_state)
        elif isinstance(robot_state, RobotState):
            self._arm_planning_component.set_start_state(robot_state=robot_state)
        else:
            msg = (
                f"robot_state must be a string or a RobotState, got {type(robot_state)}"
            )
            raise TypeError(
                msg,
            )

    def get_start_state(self) -> RobotState:
        """Get the start state."""
        return self._arm_planning_component.get_start_state()

    def plan(self) -> MotionPlanResponse:
        """Plan a trajectory to the goal."""
        return self._arm_planning_component.plan()

    def execute(self, trajectory: RobotTrajectory):
        """Execute a trajectory."""
        self._moveit_py.execute(trajectory, controllers=[])

    # TODO: Add a function to set the gripper/arm joint positions
