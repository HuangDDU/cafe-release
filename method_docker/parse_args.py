import json
import argparse

import scanpy as sc


def parse_args():
    parser = argparse.ArgumentParser(description="Parse input arguments for the analysis.")
    parser.add_argument("--adata_path", type=str, default="/data/adata.h5ad", help="Path to the adata file to be read.")
    parser.add_argument("--prior_information", type=str, default="/data/prior_information.json", help="JSON file name for prior information.")
    parser.add_argument("--parameters", type=str, default="/data/parameters.json", help="JSON file name for parameters.")
    parser.add_argument("--output_filename", type=str, default="/data/output.pkl", help="Output filename.")
    args = parser.parse_args()

    adata = sc.read(args.adata_path)
    with open(args.prior_information, 'r') as prior_file:
        prior_information = json.load(prior_file)

    with open(args.parameters, 'r') as params_file:
        parameters = json.load(params_file)

    return adata, prior_information, parameters, args.output_filename
