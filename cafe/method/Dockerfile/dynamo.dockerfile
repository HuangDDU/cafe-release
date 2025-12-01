FROM python:3.10.18

ARG CellFateExplorer

# git installation if needed

ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
COPY requirement.dynamo.txt /code/requirement.dynamo.txt
RUN pip install -r /code/requirement.dynamo.txt

COPY run.py method_decorator.py preprocess_pipeline.py cf_dynamo.py /code/
