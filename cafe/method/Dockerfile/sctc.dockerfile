FROM python:3.10.15

ARG CellFateExplorer

RUN pip install scanpy sctc igraph

COPY run.py method_decorator.py cf_sctc.py /code/
