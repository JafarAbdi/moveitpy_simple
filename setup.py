from setuptools import setup

package_name = "moveitpy_simple"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            [f"resource/{package_name}"],
        ),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="juruc",
    maintainer_email="jafar.uruc@gmail.com",
    description="A simple interface to MoveIt 2's python binding",
    license="BSD",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [],
    },
)
