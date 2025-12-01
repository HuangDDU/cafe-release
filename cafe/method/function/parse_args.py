import argparse
import importlib
import json
import logging
import os
import pickle

import scanpy as sc


def parse_args():
    parser = argparse.ArgumentParser(description="Parse input arguments for the analysis.")
    parser.add_argument("--function_name", type=str, default="cf_paga", help="Function to be executed.")
    parser.add_argument("--adata_path", type=str, default="/data/adata.h5ad", help="Path to the adata file to be read.")
    parser.add_argument("--parameters", type=str, default="/data/parameters.json", help="JSON file name for parameters.")
    parser.add_argument("--output_filename", type=str, default="/data/output.pkl", help="Output filename.")
    parser.add_argument("--save_h5ad", type=str, default=None, help="Output filename.")
    args = parser.parse_args()

    function_name = args.function_name
    adata_path = args.adata_path
    parameters = args.parameters
    output_filename = args.output_filename
    save_h5ad = args.save_h5ad
    return function_name, adata_path, parameters, output_filename, save_h5ad


def main():
    import sys

    print(f"python interpreter path: {sys.executable}")

    # optimize logging output
    os.environ["TQDM_DISABLE"] = "1"
    logging.basicConfig(level=logging.WARNING, format="%(message)s", datefmt="%S")

    # parse args from command line
    function_name, adata_path, parameters, output_filename, save_h5ad = parse_args()

    # load data, parameter, method
    # load the AnnData object as data
    adata = sc.read(adata_path)
    # load paramter dict
    with open(parameters, "r") as params_file:
        parameters = json.load(params_file)
    # load method function object
    module = importlib.import_module(f"cf_{function_name}")
    func = getattr(module, function_name)

    # execute method function
    trajectory_dict = func(adata, **parameters)

    # save result
    with open(output_filename, "wb") as f:
        pickle.dump(trajectory_dict, f)
    if save_h5ad is not None:
        if "save_h5ad" in trajectory_dict:
            # method directly save h5ad result, like pyrovelocity
            import shutil

            shutil.copyfile(trajectory_dict["save_h5ad"], save_h5ad)
            print(f"copy adata h5ad file from {trajectory_dict['save_h5ad']} to {save_h5ad}")
        else:
            print(f"save adata h5ad file to '{save_h5ad}'.")
            adata.write_h5ad(save_h5ad)
    else:
        print("'save_h5ad' is None, skip saving adata h5ad file.")
    print(f"{function_name} finish!")


# local execute:
#    python ./method/function/parse_args.py \
#         --function_name comp1 \
#         --adata_path ./method_docker_input/adata.h5ad \
#         --parameters ./method_docker_input/parameters.json \
#         --output_filename ./method_docker_output/output.pkl
if __name__ == "__main__":
    main()
