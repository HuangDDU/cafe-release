# import anndata as ad
# import scvelo as scv

# try:
#     # for docker
#     from method_decorator import method_info
#     from preprocess_pipeline import preprocess_pipeline
# except ImportError:
#     # for completed cfe environment
#     from cfe.method.function.method_decorator import method_info
#     from cfe.method.function.preprocess_pipeline import preprocess_pipeline


# @method_info(
#     name="dynamo",
#     version="0.0.1",
#     description="Dynamo: Representation learning of RNA velocity reveals robust cell transitions",
#     wrapper_type="velocity",
#     doi="10.1073/pnas.2105859118",
#     github_url="https://github.com/qiaochen/VeloAE",
# )
# def dynamo():
#     return None
