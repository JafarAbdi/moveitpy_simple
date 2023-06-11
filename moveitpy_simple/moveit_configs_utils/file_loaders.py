import math
from pathlib import Path
from tempfile import NamedTemporaryFile

import jinja2
import xacro
import yaml


class FileNotFoundError(KeyError):
    pass


def render_template(template: Path, mappings: dict):
    with template.open("r") as file:
        jinja2_template = jinja2.Template(file.read())
        jinja2_template.globals["radians"] = math.radians
        jinja2_template.globals["degrees"] = math.degrees
    return jinja2_template.render(mappings)


def raise_if_file_not_found(file_path: Path):
    if not file_path.exists():
        msg = f"File {file_path} doesn't exist"
        raise FileNotFoundError(msg)


def create_file_from_template(file: Path, mappings: dict | None = None):
    with NamedTemporaryFile(
        mode="w", prefix="moveitpy_simple_", delete=False,
    ) as parsed_file:
        parsed_file_path = parsed_file.name
        parsed_file.write(render_template(file, mappings or {}))
        return parsed_file_path


def load_file(file_path: Path, mappings: dict | None = None):
    raise_if_file_not_found(file_path)
    if mappings is not None:
        return render_template(file_path, mappings)
    try:
        with open(file_path) as file:
            return file.read()
    except (
        OSError
    ):  # parent of IOError, OSError *and* WindowsError where available
        return None


def load_yaml(file_path: Path, mappings: dict | None = None):
    raise_if_file_not_found(file_path)

    try:
        return yaml.load(
            render_template(file_path, mappings or {}), Loader=yaml.FullLoader,
        )
    except (
        OSError
    ):  # parent of IOError, OSError *and* WindowsError where available
        return None


def load_xacro(file_path: Path, mappings: dict | None = None):
    raise_if_file_not_found(file_path)

    file = xacro.process_file(file_path, mappings=mappings)
    return file.toxml()
