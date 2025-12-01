FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-runtime

ARG CellFateExplorer

RUN pip install scanpy cytotrace2-py

# copy model file
COPY cytotrace2_py/resources /opt/conda/lib/python3.11/site-packages/cytotrace2_py/resources

COPY run.py method_decorator.py cf_cytotrace2.py /code/

# TODO: GPU support
