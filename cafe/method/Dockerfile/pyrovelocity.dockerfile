# FROM python:3.11.13
# need python version>=3.11
FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-runtime

ARG CellFateExplorer

RUN apt-get update && apt-get install -y git
RUN pip install git+https://github.com/pinellolab/pyrovelocity@v0.4.5 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install numpy==1.26.4 anndata==0.10.9 scvi-tools==1.1.1 flytekit==1.13.3 flax==0.8.1 jaxlib==0.4.27 jax==0.4.27 mlflow==2.13.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install dulwich -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY run.py method_decorator.py preprocess_pipeline.py cf_pyrovelocity.py /code/