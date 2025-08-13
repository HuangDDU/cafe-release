# Installation

## Using pip(TODO)

> The installation will be simplified when Pypi package is released in the near future.

## Using github project

1. Clone the repository and enter the directory.
2. Create a conda environment and install the dependencies by running the following commands.

    ```bash
    conda create -n cfe python=3.10.15
    conda activate cfe
    pip install -r requirements.txt
    ```

3. Add the now working dir into python package path.
   - For Linux (such as working dir: /home/huang/CellFateExplorer)

     ```bash
     export PYTHONPATH="$PYTHONPATH:/home/haung/CellFateExplorer"
     ```

   - For Windows (such as working dir: D:\CellFateExplorer)

     ```cmd
     setx PYTHONPATH "%PYTHONPATH%;D:\CellFateExplorer"
     ```

   - If you use VSCode, you should create `.env` in working dir for jupyter notebook python package search as following.

     ```txt
     PYTHONPATH=/home/haung/CellFateExplorer
     ```

4. Other installation requirements:
   - R and rpy2 are required.
   - If you want to use docker container as backend, you need to install docker beforehand.
   - install conda package

        ```bash
        conda install pygraphviz
        ```
