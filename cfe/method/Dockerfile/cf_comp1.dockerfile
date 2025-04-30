FROM python:3.10.15

ARG CellFateExplorer

RUN pip install scanpy

COPY run.py cf_comp1.py definition.yml /code/