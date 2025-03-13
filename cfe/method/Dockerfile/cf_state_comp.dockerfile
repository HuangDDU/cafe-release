FROM python:3.10.15

ARG CellFateExplorer

RUN pip install scanpy

COPY run.py parse_args.py definition.yml /code/