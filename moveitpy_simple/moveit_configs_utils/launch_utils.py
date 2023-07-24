"""Utilities for launch files."""

from dataclasses import make_dataclass
from functools import wraps

from launch.actions import OpaqueFunction
from launch.launch_context import LaunchContext


def launch_configurations(func):  # noqa: ANN001, ANN201
    """Decorator to pass launch configurations to a function.

    The function must take a single argument, which is a dataclass with the launch configurations as attributes.
    """

    def args_to_dataclass(context: LaunchContext, *args, **kwargs):  # noqa: ANN202
        launch_configurations = make_dataclass(
            "LAUNCH_CONFIGURATIONS",
            context.launch_configurations.keys(),
            slots=True,
            frozen=True,
        )
        return func(launch_configurations(**context.launch_configurations))

    @wraps(func)
    def wrapper(*args, **kwargs):  # noqa: ANN202
        return [OpaqueFunction(function=args_to_dataclass)]

    return wrapper
