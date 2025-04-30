FROM python:3.10.15

ARG CellFateExplorer

RUN pip install scanpy scvelo igraph

COPY run.py cf_paga.py definition.yml /code/