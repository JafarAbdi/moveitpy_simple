"""Utility functions for loading files and parsing them with jinja2."""

import math
from copy import deepcopy
from pathlib import Path
from tempfile import NamedTemporaryFile

import jinja2
import xacro
import yaml


def render_template(template: Path, mappings: dict) -> str:
    """Render a jinja2 template with the given mappings."""
    with template.open("r") as file:
        jinja2_template = jinja2.Template(file.read())
        jinja2_template.globals["radians"] = math.radians
        jinja2_template.globals["degrees"] = math.degrees
    return jinja2_template.render(mappings)


def raise_if_file_not_found(file_path: Path) -> None:
    """Raise a FileNotFoundError if the given file doesn't exist."""
    if not file_path.exists():
        msg = f"File {file_path} doesn't exist"
        raise FileNotFoundError(msg)


def create_file_from_template(file: Path, mappings: dict | None = None) -> str:
    """Create a temporary file from a template and return the path to the created file."""
    with NamedTemporaryFile(
        mode="w",
        prefix="moveitpy_simple_",
        delete=False,
    ) as parsed_file:
        parsed_file_path = parsed_file.name
        parsed_file.write(render_template(file, mappings or {}))
        return parsed_file_path


def load_file(file_path: Path, mappings: dict | None = None) -> str | None:
    """Load a file and render it with the given mappings."""
    raise_if_file_not_found(file_path)
    if mappings is not None:
        return render_template(file_path, mappings)
    try:
        with file_path.open() as file:
            return file.read()
    except OSError:  # parent of IOError, OSError *and* WindowsError where available
        return None


def load_yaml(file_path: Path, mappings: dict | None = None) -> dict | None:
    """Load a yaml file and render it with the given mappings."""
    raise_if_file_not_found(file_path)

    try:
        return yaml.safe_load(render_template(file_path, mappings or {}))
    except OSError:  # parent of IOError, OSError *and* WindowsError where available
        return None


def load_xacro(file_path: Path, mappings: dict | None = None) -> str:
    """Load a xacro file and render it with the given mappings."""
    raise_if_file_not_found(file_path)

    # We need to deepcopy the mappings because xacro.process_file modifies them
    file = xacro.process_file(
        file_path,
        mappings=deepcopy(mappings) if mappings else {},
    )
    return file.toxml()
