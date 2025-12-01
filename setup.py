from setuptools import find_packages, setup

setup(
    name="cafe-release",
    version="0.2.0",
    author="Zhaoyang Huang",
    author_email="hzy554598474@163.com",
    description="A command line interface for Cafe (Cellular Fate Explorer) project",
    packages=find_packages(),
    install_requires=[
        "argparse",  # Add other dependencies as needed
    ],
    entry_points={
        "console_scripts": [
            "cafe=cafe.cli:main",  # Maps the command "cafe" to the main function in cli.py
        ],
    },
)
