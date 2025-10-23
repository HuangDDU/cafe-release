FROM python:3.10.18

ARG CellFateExplorer

RUN apt-get update && apt-get install -y git

ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
COPY requirement.veloae.txt /code/requirement.veloae.txt
RUN pip install -r /code/requirement.veloae.txt

COPY run.py method_decorator.py preprocess_pipeline.py cf_veloae.py /code/
