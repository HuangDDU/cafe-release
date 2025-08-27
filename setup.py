from setuptools import find_packages, setup

setup(
    name="cfe",
    version="0.1.0",
    author="Zhaoyang Huang",
    author_email="hzy554598474@163.com",
    description="A command line interface for cfe project",
    packages=find_packages(),
    install_requires=[
        "argparse",  # Add other dependencies as needed
    ],
    entry_points={
        "console_scripts": [
            "cfe=cfe.cli:main",  # Maps the command "cfe" to the main function in cli.py
        ],
    },
)
