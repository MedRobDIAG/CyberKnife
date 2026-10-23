import os
from glob import glob
from setuptools import setup

package_name = "cyberknife_ros"


def _files(sub, pattern="*"):
    return (os.path.join("share", package_name, sub),
            [p for p in glob(os.path.join(sub, pattern)) if os.path.isfile(p)])


setup(
    name=package_name,
    version="1.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        _files("urdf"),
        _files("launch"),
        _files("rviz"),
        _files("plans"),
        _files("meshes"),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Nikke03",
    maintainer_email="dolceroma54@gmail.com",
    description="Cinematica ed esecuzione del piano CyberKnife (Fase 3).",
    license="MIT",
    entry_points={
        "console_scripts": [
            "treatment_node = cyberknife_ros.treatment_node:main",
        ],
    },
)
