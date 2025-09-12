FROM python:3.10.15

ARG CellFateExplorer

RUN pip install scanpy

COPY run.py comp1.py /code/
