FROM python:3.10.14

ARG CellFateExplorer

ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
COPY requirement.stavia.txt /code/requirement.stavia.txt
RUN pip install -r /code/requirement.stavia.txt

COPY run.py method_decorator.py preprocess_pipeline.py cf_stavia.py /code/
