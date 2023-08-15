"""Test the MoveItConfigsBuilder class."""

import os
import re
from pathlib import Path

import pytest

from moveitpy_simple.moveit_configs_utils import MoveItConfigsBuilder
from moveitpy_simple.moveit_configs_utils.moveit_configs_builder import (
    ConfigSections,
    PackageNotFoundError,
    extend_configs,
    get_missing_configs,
    load_moveit_configs_toml,
)

# Get robot name from moveti_resources_ROBOT_moveit_config package name
ROBOT_NAME = re.compile(r"moveit_resources_(.*)_moveit_config")
dir_path = Path(os.path.realpath(__file__)).parent


def test_moveit_resources_configs():
    """Test the MoveItConfigsBuilder for moveit_resources's panda and fanuc moveit config packages."""
    for pkg_name in [
        "moveit_resources_fanuc_moveit_config",
        "moveit_resources_panda_moveit_config",
    ]:
        try:
            robot_name = re.match(ROBOT_NAME, pkg_name)[1]
            builder = MoveItConfigsBuilder(pkg_name)
            assert builder._robot_description_config is None
            assert builder._robot_description_semantic_config is None
            assert builder._robot_description_kinematics_config is None
            assert builder._planning_pipelines_config is None
            assert builder._trajectory_execution_config is None
            assert builder._sensors_config is None
            assert builder._joint_limits_config is None
            assert builder._moveit_cpp_config is None
            builder.robot_description(file_path=f"config/{robot_name}.urdf.xacro")
            builder.robot_description_semantic(file_path=f"config/{robot_name}.srdf")
            builder.robot_description_kinematics(file_path="config/kinematics.yaml")
            builder.planning_pipelines(
                pipelines=[
                    "ompl",
                    "chomp",
                ],
            )
            builder.trajectory_execution(file_path="config/moveit_controllers.yaml")
            if robot_name == "panda":
                builder.sensors(file_path="config/sensors_kinect_pointcloud.yaml")
            else:
                builder.sensors(file_path="config/sensors_3d.yaml")
            builder.joint_limits(file_path="config/joint_limits.yaml")

            with pytest.raises(FileNotFoundError):
                builder.moveit_cpp(file_path="config/moveit_cpp.yaml")

            assert builder._robot_description_config is not None
            assert builder._robot_description_semantic_config is not None
            assert builder._robot_description_kinematics_config is not None
            assert len(builder._planning_pipelines_config.pipelines) == 2
            assert len(builder._planning_pipelines_config.configs) == 2
            assert builder._trajectory_execution_config is not None
            assert builder._sensors_config is not None
            assert builder._joint_limits_config is not None
            assert builder._moveit_cpp_config is None

            assert builder.to_moveit_configs()

        except RuntimeError as e:
            msg = f"Default {pkg_name} configuration failed to build: {e}"
            raise AssertionError(msg) from e


def test_panda():
    """Test loading a robot package by specifying its name."""
    builder = (
        MoveItConfigsBuilder(package=Path(dir_path, "panda_moveit_config"))
        .robot_description()
        .robot_description_semantic()
        .robot_description_kinematics()
        .planning_pipelines()
        .trajectory_execution()
        .sensors()
        .joint_limits()
    )
    assert (moveit_configs := builder.to_moveit_configs()) is not None

    builder = (
        MoveItConfigsBuilder(package=Path(dir_path, "extend_local_panda_moveit_config"))
        .robot_description()
        .robot_description_semantic()
        .robot_description_kinematics()
        .planning_pipelines()
        .trajectory_execution()
        .sensors()
        .joint_limits()
    )

    assert (moveit_configs := builder.to_moveit_configs()) is not None
    assert (
        moveit_configs.robot_description_kinematics["robot_description_kinematics"][
            "panda_arm"
        ]["kinematics_solver"]
        == "pick_ik/PickIkPlugin"
    )


def test_robot():
    """Test loading a robot package by specifying its path."""
    builder = (
        MoveItConfigsBuilder(package=Path(dir_path, "robot_moveit_config"))
        .robot_description()
        .robot_description_semantic()
        .robot_description_kinematics()
        .planning_pipelines()
        .trajectory_execution()
        .joint_limits()
        .sensors()
    )
    assert builder._robot_description_config is not None
    assert builder._robot_description_semantic_config is not None
    assert builder._robot_description_kinematics_config is not None
    assert builder._planning_pipelines_config is not None
    assert builder._trajectory_execution_config is not None
    assert builder._sensors_config is not None
    assert builder._robot_description_config.mappings == {
        "robot_name": "kermit",
        "robot_model_type": "urdf",
    }
    assert len(builder._planning_pipelines_config.pipelines) == 4
    for pipeline_name, pipeline_config in zip(
        builder._planning_pipelines_config.pipelines,
        builder._planning_pipelines_config.configs,
        strict=True,
    ):
        if pipeline_name == "ompl":
            assert pipeline_config.mappings == {"group_name": "arm"}
        elif pipeline_name == "chomp":
            assert pipeline_config.mappings == {"group_type": "chain"}
    assert builder.to_moveit_configs()

    # Test passing in mappings to the robot description config without overriding the file path
    builder = MoveItConfigsBuilder(
        package=Path(dir_path, "robot_moveit_config"),
    ).robot_description(mappings={"robot": "myrobot", "model": "urdf"})
    assert builder._robot_description_config is not None
    assert builder._robot_description_config.mappings == {
        "robot": "myrobot",
        "model": "urdf",
    }
    assert builder.to_moveit_configs()


def test_extend():
    """Test extending a MoveItConfigsBuilder with additional configs from a different package."""
    robot_package_path = Path(dir_path, "robot_moveit_config")
    configs = load_moveit_configs_toml(robot_package_path)
    assert get_missing_configs(configs) == [
        ConfigSections.MOVEIT_CPP,
        ConfigSections.PILZ_CARTESIAN_LIMITS,
    ]

    robot2_package_path = Path(dir_path, "robot2_moveit_config")
    configs = load_moveit_configs_toml(robot2_package_path)
    assert len(get_missing_configs(configs)) == 5

    configs = extend_configs(robot2_package_path, configs)
    assert get_missing_configs(configs) == [
        ConfigSections.MOVEIT_CPP,
        ConfigSections.PILZ_CARTESIAN_LIMITS,
    ]
    moveit_configs = configs[ConfigSections.MOVEIT_CONFIGS]
    assert moveit_configs[ConfigSections.ROBOT_DESCRIPTION] == "config/kermit2.urdf"
    assert (
        moveit_configs[ConfigSections.ROBOT_DESCRIPTION_SEMANTIC].resolve()
        == robot_package_path / "config" / "kermit.srdf"
    )
    assert (
        moveit_configs[ConfigSections.ROBOT_DESCRIPTION_KINEMATICS]
        == "config/kinematics2.yaml"
    )
    assert moveit_configs[ConfigSections.PLANNING_PIPELINES] == {
        "ompl": "config/ompl_planning2.yaml",
    }
    assert moveit_configs[ConfigSections.JOINT_LIMITS] == "config/joint_limits2.yaml"
    assert (
        moveit_configs[ConfigSections.TRAJECTORY_EXECUTION].resolve()
        == robot_package_path / "config" / "moveit_controllers.yaml"
    )
    assert (
        moveit_configs[ConfigSections.SENSORS].resolve()
        == robot_package_path / "config" / "sensors_kinect_pointcloud.yaml"
    )
    assert configs[ConfigSections.ROBOT_DESCRIPTION] == {"robot_name": "kermit2"}
    assert configs.get(ConfigSections.PLANNING_PIPELINES) is None
    assert configs[ConfigSections.ROBOT_DESCRIPTION_SEMANTIC] == {"group_type": "chain"}

    robot3_package_path = Path(dir_path, "robot3_moveit_config")
    configs = load_moveit_configs_toml(robot3_package_path)
    assert len(get_missing_configs(configs)) == 8

    configs = extend_configs(robot3_package_path, configs)
    assert get_missing_configs(configs) == [
        ConfigSections.MOVEIT_CPP,
        ConfigSections.PILZ_CARTESIAN_LIMITS,
    ]
    moveit_configs = configs[ConfigSections.MOVEIT_CONFIGS]
    assert moveit_configs[ConfigSections.ROBOT_DESCRIPTION] == "config/kermit3.urdf"
    assert (
        moveit_configs[ConfigSections.ROBOT_DESCRIPTION_SEMANTIC].resolve()
        == robot_package_path / "config" / "kermit.srdf"
    )
    assert (
        moveit_configs[ConfigSections.ROBOT_DESCRIPTION_KINEMATICS].resolve()
        == robot2_package_path / "config" / "kinematics2.yaml"
    )
    assert len(moveit_configs[ConfigSections.PLANNING_PIPELINES]) == 1
    assert (
        moveit_configs[ConfigSections.PLANNING_PIPELINES]["ompl"].resolve()
        == robot2_package_path / "config" / "ompl_planning2.yaml"
    )
    assert (
        moveit_configs[ConfigSections.JOINT_LIMITS].resolve()
        == robot2_package_path / "config" / "joint_limits2.yaml"
    )
    assert (
        moveit_configs[ConfigSections.TRAJECTORY_EXECUTION].resolve()
        == robot_package_path / "config" / "moveit_controllers.yaml"
    )
    assert (
        moveit_configs[ConfigSections.SENSORS].resolve()
        == robot_package_path / "config" / "sensors_kinect_pointcloud.yaml"
    )
    assert configs.get(ConfigSections.ROBOT_DESCRIPTION) is None
    assert configs.get(ConfigSections.PLANNING_PIPELINES) is None
    assert configs[ConfigSections.ROBOT_DESCRIPTION_SEMANTIC] == {"group_type": "chain"}


def test_extend_builder():
    """Test extending a MoveItConfigsBuilder with additional configs from a different package and creating MoveItConfigs class."""
    builder = (
        MoveItConfigsBuilder(package=Path(dir_path, "robot2_moveit_config"))
        .robot_description()
        .robot_description_semantic()
        .robot_description_kinematics()
        .planning_pipelines()
        .trajectory_execution()
        .joint_limits()
        .sensors()
    )
    moveit_configs = builder.to_moveit_configs()
    assert moveit_configs.robot_description
    assert moveit_configs.robot_description_semantic
    assert moveit_configs.robot_description_kinematics
    assert moveit_configs.planning_pipelines
    assert moveit_configs.trajectory_execution
    assert moveit_configs.joint_limits
    assert moveit_configs.sensors_3d

    builder = (
        MoveItConfigsBuilder(package=Path(dir_path, "robot3_moveit_config"))
        .robot_description()
        .robot_description_semantic()
        .robot_description_kinematics()
        .planning_pipelines()
        .trajectory_execution()
        .joint_limits()
        .sensors()
    )
    assert builder.to_moveit_configs()
    assert moveit_configs.robot_description
    assert moveit_configs.robot_description_semantic
    assert moveit_configs.robot_description_kinematics
    assert moveit_configs.planning_pipelines
    assert moveit_configs.trajectory_execution
    assert moveit_configs.joint_limits
    assert moveit_configs.sensors_3d


def test_extend_panda():
    """Test extending a MoveItConfigsBuilder with additional configs from a named package."""
    builder = (
        MoveItConfigsBuilder(package=Path(dir_path, "panda_extend_moveit_config"))
        .robot_description(
            file_path=Path(dir_path, "robot.urdf.xacro"),
            mappings={
                "test_robot": "my_robot",
                "postfix": "my_postfix",
                "number": "5",
            },
        )
        .robot_description_kinematics()
    )

    with pytest.raises(RuntimeError):
        builder.robot_description_semantic()

    moveit_configs = builder.to_moveit_configs()
    assert moveit_configs.robot_description
    assert moveit_configs.robot_description_semantic == {}
    assert (
        moveit_configs.robot_description_kinematics["robot_description_kinematics"][
            "panda_arm"
        ]["kinematics_solver"]
        == "pick_ik/PickIkPlugin"
    )


def test_extend_non_existing_package():
    """Test loading configs from a non-existing packages."""
    with pytest.raises(PackageNotFoundError):
        MoveItConfigsBuilder(
            package=Path(dir_path, "extend_non_existing_moveit_config"),
        )
    with pytest.raises(PackageNotFoundError):
        MoveItConfigsBuilder(package="non_existing_package")
    with pytest.raises(PackageNotFoundError):
        MoveItConfigsBuilder(package=Path(dir_path, "existing_moveit_config"))


def test_load_all():
    """Test automatically loading all configs from a package."""
    builder = MoveItConfigsBuilder(
        package=Path(dir_path, "partial_panda_moveit_config"),
    ).load_all()
    assert (moveit_configs := builder.to_moveit_configs())
    assert moveit_configs.robot_description
    assert moveit_configs.robot_description_kinematics
    assert moveit_configs.joint_limits
    assert moveit_configs.planning_pipelines
    assert (
        len(moveit_configs.planning_pipelines["planning_pipelines.pipeline_names"]) == 2
    )
    assert moveit_configs.trajectory_execution

    assert not moveit_configs.robot_description_semantic
    assert not moveit_configs.sensors_3d
    assert not moveit_configs.pilz_cartesian_limits
    assert not moveit_configs.moveit_cpp

    builder = MoveItConfigsBuilder(
        package=Path(dir_path, "panda_moveit_config"),
    ).load_all()
    assert (moveit_configs := builder.to_moveit_configs())
    assert moveit_configs.robot_description
    assert moveit_configs.robot_description_semantic
    assert moveit_configs.robot_description_kinematics
    assert moveit_configs.joint_limits
    assert moveit_configs.planning_pipelines
    assert (
        len(moveit_configs.planning_pipelines["planning_pipelines.pipeline_names"]) == 2
    )
    assert moveit_configs.trajectory_execution
    assert moveit_configs.sensors_3d

    assert not moveit_configs.pilz_cartesian_limits


def test_load_with_filename():
    """Test loading the moveit config from a specific file."""
    builder = MoveItConfigsBuilder(
        package=Path(dir_path, "custom_moveit_config.toml"),
    ).load_all()
    assert (moveit_configs := builder.to_moveit_configs())
    assert moveit_configs.robot_description
    assert moveit_configs.robot_description_semantic
    assert moveit_configs.robot_description_kinematics
    assert moveit_configs.joint_limits
    assert moveit_configs.planning_pipelines
    assert (
        len(moveit_configs.planning_pipelines["planning_pipelines.pipeline_names"]) == 1
    )
    assert moveit_configs.trajectory_execution
    assert not moveit_configs.sensors_3d
    assert not moveit_configs.pilz_cartesian_limits
    assert not moveit_configs.moveit_cpp
    assert not builder._robot_description_config.mappings

    builder = MoveItConfigsBuilder(
        package=Path(dir_path, "extend_custom_moveit_config.toml"),
    ).load_all()
    assert (moveit_configs := builder.to_moveit_configs())
    assert moveit_configs.robot_description
    assert moveit_configs.robot_description_semantic
    assert moveit_configs.robot_description_kinematics
    assert moveit_configs.joint_limits
    assert moveit_configs.planning_pipelines
    assert (
        len(moveit_configs.planning_pipelines["planning_pipelines.pipeline_names"]) == 1
    )
    assert moveit_configs.trajectory_execution
    assert not moveit_configs.sensors_3d
    assert not moveit_configs.pilz_cartesian_limits
    assert not moveit_configs.moveit_cpp
    assert builder._robot_description_config.mappings == {
        "robot_name": "panda",
        "robot_model_type": "urdf.xacro",
    }
