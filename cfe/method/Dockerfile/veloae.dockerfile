FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

ARG CellFateExplorer

RUN apt-get update && apt-get install -y git
RUN pip install scipy==1.10.1 numpy==1.23.5 pandas==1.3.5 matplotlib==3.7.5 anndata==0.9.2 scanpy==1.9.6 scvelo==0.2.5 torch-geometric 
RUN pip install git+https://github.com/qiaochen/VeloAE

COPY run.py method_decorator.py preprocess_pipeline.py cf_veloae.py /code/
