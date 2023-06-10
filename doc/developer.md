# Developers Guide

To make sure you have the latest repos:

```bash
cd $COLCON_WS/src/moveitpy_simple
git checkout main
git pull origin main
cd $COLCON_WS/src
vcs import < moveitpy_simple/moveitpy_simple.repos
rosdep install --from-paths . --ignore-src -y
```

## Setup pre-commit

pre-commit is a tool to automatically run formatting checks on each commit, which saves you from manually running clang-format (or, crucially, from forgetting to run them!).

Install pre-commit like this:

```
pip3 install pre-commit
```

Run this in the top directory of the repo to set up the git hooks:

```
pre-commit install
```

## Testing and Linting

To test the packages in moveitpy_simple, use the following command with [colcon](https://colcon.readthedocs.io/en/released/).

```bash
export TEST_PACKAGES="PROJECT_PACKAGE_NAMES"
colcon build --packages-up-to ${TEST_PACKAGES}
colcon test --packages-select ${TEST_PACKAGES}
colcon test-result
```
