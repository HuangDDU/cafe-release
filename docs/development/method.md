# Method

> This article will use the addition of a classic velocity wrapper `scvelo` as an example to illustrate how developers can contribute new trajectory inference methods to this project. You should develop new trajectory inference method in `cfe/method` directory.

## Register fate method and create environment

1. ensure you have local python environment ``
2. `cfe/method/definition/cf_scvelo.yml`
3. `cfe/method/method_backend.yml`

## Python function

1. `cfe/method/function/cf_scvelo.py`

## Test in jupyter notebook

1. Test in pytest script:
2. Test in jupyter notbeook: `notebook_dev/hzy/dev_scvelo_pancreas_500_conda.ipynb`

## Submit conda environment(TODO)

## Docker environment(TODO)
