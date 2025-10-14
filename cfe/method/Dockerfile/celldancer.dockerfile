# FROM python:3.11.13
# need python version>=3.11
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

ARG CellFateExplorer

RUN apt-get update && apt-get install -y git
RUN pip install git+https://github.com/GuangyuWangLab2021/cellDancer.git -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install scanpy scvelo -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY run.py method_decorator.py preprocess_pipeline.py cf_celldancer.py /code/