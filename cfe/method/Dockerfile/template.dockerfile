FROM python:$python_version

ARG CellFateExplorer

ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
COPY requirement.$method_name.txt /code/requirement.$method_name.txt
RUN pip install -r /code/requirement.$method_name.txt

COPY run.py method_decorator.py preprocess_pipeline.py cf_$method_name.py /code/
