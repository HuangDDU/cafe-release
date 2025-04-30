FROM python:3.10.15

ARG CellFateExplorer

RUN pip install scanpy

COPY run.py cf_state_comp.py definition.yml /code/