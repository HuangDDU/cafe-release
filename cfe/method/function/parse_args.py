import argparse
import importlib
import json
import pickle

import scanpy as sc


def parse_args():
    parser = argparse.ArgumentParser(description="Parse input arguments for the analysis.")
    parser.add_argument("--function_name", type=str, default="cf_paga", help="Function to be executed.")
    parser.add_argument("--adata_path", type=str, default="/data/adata.h5ad", help="Path to the adata file to be read.")
    parser.add_argument(
        "--prior_information",
        type=str,
        default="/data/prior_information.json",
        help="JSON file name for prior information.",
    )
    parser.add_argument("--parameters", type=str, default="/data/parameters.json", help="JSON file name for parameters.")
    parser.add_argument("--output_filename", type=str, default="/data/output.pkl", help="Output filename.")
    args = parser.parse_args()

    function_name = args.function_name
    adata = sc.read(args.adata_path)
    with open(args.prior_information, "r") as prior_file:
        prior_information = json.load(prior_file)

    with open(args.parameters, "r") as params_file:
        # TODO: update from yml parameter dict
        parameters = json.load(params_file)

    return function_name, adata, prior_information, parameters, args.output_filename


# local execute:
#    python ./cfe/method/function/parse_args.py \
#         --function_name cf_paga \
#         --adata_path ./method_docker_input/adata.h5ad \
#         --prior_information ./method_docker_input/prior_information.json \
#         --parameters ./method_docker_input/parameters.json \
#         --output_filename ./method_docker_output/output.pkl
if __name__ == "__main__":
    function_name, adata, prior_information, parameters, output_filename = parse_args()

    module = importlib.import_module(f"{function_name}")
    func = getattr(module, function_name)
    trajectory_dict = func(adata, prior_information, parameters)
    with open(output_filename, "wb") as f:
        pickle.dump(trajectory_dict, f)
    print(f"{function_name} finish!")
