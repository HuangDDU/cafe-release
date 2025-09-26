FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-runtime

ARG CellFateExplorer

RUN pip install scanpy scvelo

COPY run.py method_decorator.py preprocess_pipeline.py cf_scvelo.py /code/