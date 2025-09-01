FROM python:3.10.15

ARG CellFateExplorer

RUN pip install scanpy scvelo

COPY run.py cf_scvelo.py definition.yml /code/
