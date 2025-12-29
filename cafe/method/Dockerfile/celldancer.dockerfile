FROM python:3.7.16

ARG CellFateExplorer

# git installation if needed

ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
COPY requirement.celldancer.txt /code/requirement.celldancer.txt
RUN pip install -r /code/requirement.celldancer.txt

COPY run.py method_decorator.py preprocess_pipeline.py cf_celldancer.py /code/
