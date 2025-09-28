FROM python:3.10.15

ARG CellFateExplorer

RUN pip install scanpy scvelo dynamo-release

COPY run.py method_decorator.py preprocess_pipeline.py cf_dynamo.py /code/