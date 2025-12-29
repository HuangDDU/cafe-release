# Installation
## Create conda environment
  ```bash
  conda create -n cafe python=3.10.15
  conda activate cafe
  ```
## Using pip
  ```bash
  # for user
  pip install cafe-release
  # for developer, docs or others.
  pip install cafe-release[dev]
  ```

## Using github project for latest version
  ```bash
  git clone https://github.com/HuangDDU/cafe-release.git
  cd cafe-release
  pip install .
  ```

## (TODO)Using uv for faster installation


## Other installation requirements:
   - R and rpy2 are required.
   - If you want to use docker container as backend, you need to install docker beforehand.
   - install conda package

        ```bash
        conda install pygraphviz
        ```
