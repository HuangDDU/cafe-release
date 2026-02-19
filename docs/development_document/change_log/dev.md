# dev

## 1. Method

1. Velocity wrapper
      - `scv.tl.paga` is used to get milestone network.`Projection` wrapper is used to get   percentages.
     - TODO: LAP is also considerable.  
     - `scVelo` is implemented.
     - `VeloVI` is implemented with commit id `73f6ba1978a0677292cdc84b1f3933ce6ff9586f`.

2. Conda backend: for better method development adaptability
      - Significance: method developer just run successful in an alone conda environment, then it's easily to itergrate it to the project.
      - Principle: call shell command `conda run ...`.

3. `Yuchuan Peng` join this project. He is responsible for velocity methods like `veloAE`, `UnitVelo`, `Dynamo` and so on.

4. Method parameter optimization
      - Dynamical import trajectory method function in `cfe/method/__init__.py`.
      - API document .
      - Run with raytune

## Developer tools

1. Project management file: `pyproject.toml` , including project metadata, tools like pytest, flake and so on.
2. Pytest run in paralle with `pytest-xdist`.
