from pathlib import Path

import numpy as np

from moveitpy_simple.moveit_configs_utils.file_loaders import load_file, load_yaml

dir_path = Path(__file__).parent.absolute()


def test_laod_yaml():
    yaml_file = load_yaml(
        dir_path / "parameters_template.yaml",
        mappings={
            "namespace": "env_0",
            "robots": [
                {"name": "ur", "ip": "127.0.0.1"},
                {"name": "panda", "ip": "127.0.0.2"},
            ],
            "names": ["name1", "name2", "name3"],
        },
    )
    assert (env := yaml_file["env_0"]) is not None
    assert env["names"] == {"ur": "127.0.0.1", "panda": "127.0.0.2"}
    assert (radians := yaml_file["radians"]) is not None
    assert np.allclose(radians, 2.0944, atol=1e-3)
    assert (degrees := yaml_file["degrees"]) is not None
    assert np.allclose(degrees, 57.2958, atol=1e-3)
    assert len(yaml_file["names"]) == 3


def test_load_file():
    file_content = load_file(
        dir_path / "parameter_file_template", mappings={"test_name": "testing"},
    )
    assert file_content == "This's a template parameter file testing"
