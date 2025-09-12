FROM python:3.10.15

ARG CellFateExplorer

RUN pip install scanpy scvelo

COPY run.py scvelo.py /code/
