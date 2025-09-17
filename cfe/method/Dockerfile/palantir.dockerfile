FROM python:3.10.15

ARG CellFateExplorer

RUN pip install scanpy palantir

COPY run.py method_decorator.py cf_palantir.py /code/
