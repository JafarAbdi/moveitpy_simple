"""Visualization of robot states and trajectories using Panda3D."""

import time
from pathlib import Path

import numpy as np
from ament_index_python.packages import get_package_share_path
from moveit.core.robot_state import RobotState
from moveit.core.robot_trajectory import RobotTrajectory
from panda3d_viewer import Viewer, ViewerConfig
from transforms3d.euler import euler2mat
from urdf_parser_py.urdf import URDF, Box, Cylinder, Link, Mesh, Sphere

ROOT_NAME = "root"


def normalize_filename(filename: str) -> str:
    """Normalize the filename to use absolute paths."""
    if filename.startswith("package://"):
        package_name, relative_path = filename.split("package://")[1].split("/", 1)
        filename = f"/{get_package_share_path(package_name)}/{relative_path}"
    elif filename.startswith("file://"):
        filename = filename.split("file://")[1]
    return filename


def normalized_robot_description(file: Path | str) -> URDF:
    """Normalize the robot description file to use absolute paths for the collision and visual tags."""
    if isinstance(file, str):
        robot = URDF.from_xml_string(file)
    elif isinstance(file, Path):
        robot = URDF.from_xml_file(file)

    for link in robot.links:
        for collision in link.collisions:
            if isinstance(
                collision.geometry,
                Mesh,
            ):
                collision.geometry.filename = normalize_filename(
                    collision.geometry.filename,
                )

        for visual in link.visuals:
            if isinstance(
                visual.geometry,
                Mesh,
            ):
                visual.geometry.filename = normalize_filename(visual.geometry.filename)
    return robot


class Visualizer:
    """Visualize robot states and trajectories using Panda3D."""

    def __init__(self, robot_description: str, window_type: str) -> None:
        """Initialize the visualizer."""
        config = ViewerConfig()
        config.set_window_size(320, 240)
        config.enable_antialiasing(enable=True, multisamples=4)
        config.enable_shadow(enable=False)
        config.show_axes(show=False)

        # TODO: Add a way to set onscreen
        self._viewer = Viewer(window_type=window_type, config=config)

        self._viewer.append_group(ROOT_NAME)
        self._robot = normalized_robot_description(robot_description)

        for link in self._robot.links:
            if len(link.visuals) == 0:
                continue
            if len(link.visuals) != 1:
                msg = f"Only one visual per link is supported, but link {link.name} has {len(link.visuals)}"
                raise NotImplementedError(
                    msg,
                )

            visual = link.visuals[0]
            if isinstance(visual.geometry, Box):
                self._viewer.append_box(
                    ROOT_NAME,
                    link.name,
                    [
                        visual.geometry.size[0],
                        visual.geometry.size[1],
                        visual.geometry.size[2],
                    ],
                )
            elif isinstance(visual.geometry, Cylinder):
                self._viewer.append_cylinder(
                    ROOT_NAME,
                    link.name,
                    visual.geometry.radius,
                    visual.geometry.length,
                )
            elif isinstance(visual.geometry, Sphere):
                self._viewer.append_sphere(ROOT_NAME, link.name, visual.geometry.radius)
            elif isinstance(visual.geometry, Mesh):
                self._viewer.append_mesh(ROOT_NAME, link.name, visual.geometry.filename)

        self._viewer.reset_camera(pos=(1.5, 1.5, 2), look_at=(0, 0, 0.5))

    def _get_visual_origin(self, link: Link) -> np.ndarray:
        """Get the origin of the visual in the link frame."""
        if len(link.visuals) == 0:
            return np.eye(4)
        if len(link.visuals) != 1:
            msg = f"Only one visual per link is supported, but link {link.name} has {len(link.visuals)}"
            raise NotImplementedError(
                msg,
            )
        transform = np.eye(4)
        visual = link.visuals[0]
        if visual.origin is None:
            return transform
        transform[:3, :3] = euler2mat(
            visual.origin.rpy[0],
            visual.origin.rpy[1],
            visual.origin.rpy[2],
            axes="rxyz",
        )
        transform[0, 3] = visual.origin.xyz[0]
        transform[1, 3] = visual.origin.xyz[1]
        transform[2, 3] = visual.origin.xyz[2]
        return transform

    def visualize_robot_state(self, robot_state: RobotState) -> None:
        """Visualize the robot state in the viewer."""
        for link in self._robot.links:
            robot_state.robot_model.get_link_model(link.name)
            self._viewer.move_nodes(
                ROOT_NAME,
                {
                    link.name: robot_state.get_global_link_transform(link.name)
                    @ self._get_visual_origin(link),
                },
            )

    def visualize_robot_trajectory(
        self,
        robot_trajectory: RobotTrajectory | list[RobotState],
    ) -> None:
        """Visualize the robot trajectory in the viewer."""
        if isinstance(robot_trajectory, RobotTrajectory):
            for rs, duration_from_start in robot_trajectory:
                self.visualize_robot_state(rs)
                time.sleep(duration_from_start)
        elif isinstance(robot_trajectory, list):
            for rs in robot_trajectory:
                self.visualize_robot_state(rs)
                time.sleep(0.1)

    def get_robot_state_image(self, robot_state: RobotState) -> np.ndarray:
        """Get the image of the robot state, could be used for wandb."""
        self.visualize_robot_state(robot_state)
        return self._viewer.get_screenshot(requested_format="RGB")

    def get_robot_trajectory_images(
        self,
        robot_trajectory: RobotTrajectory | list[RobotState],
    ) -> np.ndarray:
        """Get the images of the robot trajectory, could be used for wandb."""
        images = []
        if isinstance(robot_trajectory, RobotTrajectory):
            images.extend(
                np.asarray(self.get_robot_state_image(rs)).transpose(2, 0, 1)
                for rs, _ in robot_trajectory
            )
        elif isinstance(robot_trajectory, list):
            images.extend(
                np.asarray(self.get_robot_state_image(rs)).transpose(2, 0, 1)
                for rs in robot_trajectory
            )
        return np.stack(images, axis=0)
