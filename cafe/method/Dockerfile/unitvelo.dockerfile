# need python version = 3.7
FROM docker.io/tensorflow/tensorflow:2.13.0

ARG CellFateExplorer

RUN apt-get update && apt-get install -y git
RUN pip install git+https://github.com/StatBiomed/UniTVelo -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install numpy==1.23.5 pandas==1.5.3 matplotlib==3.4.3 scanpy==1.5.1 scvelo==0.2.2 -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY run.py method_decorator.py preprocess_pipeline.py cf_unitvelo.py /code/