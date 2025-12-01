import argparse
import os

import pandas as pd
import scanpy as sc
import yaml

import cafe

# 暂时使用python命令调用
# python cfe/cli.py infer --data tests/data/bifurcating.h5ad --method comp1
# python cfe/cli.py infer --data tmp/input.h5ad --method comp1 --save_fig tmp/comp1.jpg --save_h5ad tmp/comp1.h5ad --parameter_file tmp/comp1.yaml
# python cfe/cli.py infer --data tmp/input.h5ad --method paga --save_fig tmp/tmp.jpg --save_h5ad tmp/output.h5ad --parameter_file tmp/paga.yaml
# python cfe/cli.py infer --data tmp/input.h5ad --method scvelo --save_fig tmp/scvelo.jpg --save_h5ad tmp/scvelo.h5ad --parameter_file tmp/scvelo.yaml

# python cfe/cli.py benchmark --data tmp/input.h5ad --method_list comp1 paga --save_fig tmp/comp1.jpg --save_h5ad tmp/comp1.h5ad --parameter_file tmp/comp1.yaml
# 后续封装好pip包之后，可以直接使用命令行调用
# TODO: cfe --data tests/data/bifurcating.h5ad --method comp1


def main():
    parser = argparse.ArgumentParser(description="Command line interface for Cafe project.")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Sub command to run")

    # infer sub command
    # python cfe/cli.py infer --data tests/data/bifurcating.h5ad --method comp1
    infer_parser = subparsers.add_parser("infer", help="Run single trajectory inference method")
    infer_parser.add_argument("-d", "--data", type=str, required=True, help="Path to the h5ad data file")
    infer_parser.add_argument("-m", "--method", type=str, required=True, choices=["comp1", "paga", "scvelo"], help="Method to execute")
    infer_parser.add_argument("--save_fig", type=str, default="tmp.jpg", help="Filename for trajecotry plot")
    infer_parser.add_argument("--save_h5ad", type=str, default=None, help="Filename for h5ad")
    infer_parser.add_argument("--parameter_file", type=str, default=None, help="Parameter yaml file for method")

    # benchmark sub command
    # python cfe/cli.py benchmark --data tests/data/bifurcating.h5ad --method_list comp1 paga
    benchmark_parser = subparsers.add_parser("benchmark", help="Run benchmark")
    benchmark_parser.add_argument("-d", "--data", type=str, required=True, help="Path to the h5ad data file")
    benchmark_parser.add_argument("--method_list", nargs="+", required=True, help="Method list to execute")
    benchmark_parser.add_argument("--metric_list", nargs="+", default=None, help="Accuracy metric list to compute")
    benchmark_parser.add_argument("--save_fig_dir", type=str, default="tmp", help="Directory of trajecotry plot")
    benchmark_parser.add_argument("--parameter_dir", type=str, default=None, help="Directory of parameter yaml for method")
    benchmark_parser.add_argument("--save_h5ad", type=str, default=None, help="Filename for h5ad")
    benchmark_parser.add_argument("--save_benchmark", default=None, help="Filename for benchmark dataframe")

    args = parser.parse_args()
    # print(args.__dict__)

    if args.command == "infer":
        infer(
            data=args.data,
            method=args.method,
            parameter_file=args.parameter_file,
            save_fig=args.save_fig,
            save_h5ad=args.save_h5ad,
        )
    elif args.command == "benchmark":
        metric_dict = benchmark(
            data=args.data,
            method_list=args.method_list,
            parameter_dir=args.parameter_dir,
            save_fig_dir=args.save_fig_dir,
            save_h5ad=args.save_h5ad,
            save_benchmark=args.save_benchmark,
        )
        print(metric_dict)


def infer(
    data,
    method,
    parameter_file,
    save_fig,
    save_h5ad,
):
    # ref to notebook/quickstart.ipynb,
    print(f"infer trajectory..., data: {data}, method: {method}")

    # data
    adata = sc.read_h5ad(data)
    fadata = cafe.data.FateAnnData.from_anndata(adata)

    # method
    if os.path.exists(parameter_file):
        parameters = yaml.safe_load(open(parameter_file, "r"))
        print(f"loading from file: {parameter_file}, parameters: {parameters}")
    else:
        parameters = {}
        print(f"parameter file don't exist: {parameter_file}, parameters: {parameters}")
    method = cafe.method.FateMethod(method_name=method)
    method.infer_trajectory(fadata, parameters=parameters)  # add parameters when inferring trajectory

    # plot
    cafe.plot.plot_trajectory(fadata, basis="umap", color=["milestone"], save=save_fig)
    print(f"{method} trajectory done, save figure to {save_fig}")

    # save
    if save_h5ad:
        fadata.write_h5ad(save_h5ad)
        print(f"save h5ad to {save_h5ad}!")

    print(f"infer trajectory done, save figure to {save_fig}, save h5ad to {save_h5ad}!")


def benchmark(
    data,
    method_list,
    parameter_dir,
    save_fig_dir,
    save_h5ad,
    save_benchmark,
):
    # TODO: refer to notebook/benchmark.ipynb
    metric_dict = {}
    # if parallel
    print(f"benchmark trajectory..., data: {data}, method_list:{method_list}")

    import cafe

    # data
    adata = sc.read_h5ad(data)
    fadata = cafe.data.FateAnnData.from_anndata(adata)

    # method
    for method_name in method_list:
        method_metric_dict = {}
        parameter_file = f"{parameter_dir}/{method_name}.yaml"
        if os.path.exists(parameter_file):
            parameters = yaml.safe_load(open(parameter_file, "r"))
            print(f"loading from file: {parameter_file}, parameters: {parameters}")
        else:
            parameters = {}
            print(f"parameter file don't exist: {parameter_file}, parameters: {parameters}")

        parameters["benchmark_resource"] = True  # show resource usage
        method = cafe.method.FateMethod(method_name=method_name, backend_name="conda")
        method.infer_trajectory(fadata, parameters=parameters)  # add parameters when inferring trajectory
        # plot
        fig_file = f"{save_fig_dir}/{method.method_name}.jpg"
        cafe.plot.plot_trajectory(fadata, basis="umap", color=["milestone"], save=fig_file)
        print(f"{method} trajectory done, save figure to {fig_file}")

        # resource usage benchmark
        resource_usage_dict = fadata.get_resource_usage()
        print(f"resource usage: {resource_usage_dict}")
        method_metric_dict["resource_usage"] = resource_usage_dict

        # TODO: accuracy metric usage benchmark
        method_metric_dict["accuracy"] = {}
        metric_dict[method_name] = method_metric_dict

    # save h5ad
    if save_h5ad:
        fadata.write_h5ad(save_h5ad)
        print(f"save h5ad to {save_h5ad}!")
    # save benchmark dataframe
    if save_benchmark:
        benchmark_df = pd.DataFrame.from_dict(metric_dict)
        benchmark_df.to_csv(save_benchmark)
    print("benchmark trajectory done")

    return metric_dict


def default_parameter_test():
    # default parameter test case
    import sys

    # # infer sub command test
    # # method_name = "comp1"
    # method_name = "scvelo"
    # sys.argv = [
    #     "cafe/cli.py", "infer",
    #     "--data", "tmp/input.h5ad",
    #     "--method", method_name,
    #     "--save_fig", f"tmp/{method_name}.jpg",
    #     "--save_h5ad", f"tmp/{method_name}.h5ad",
    #     "--parameter_file", f"tmp/{method_name}.yaml"
    # ]
    # benchmark sub command test
    sys.argv = [
        "cafe/cli.py",
        "benchmark",
        "--data",
        "tmp/input.h5ad",
        "--method_list",
        "comp1",
        "scvelo",
        "--save_fig_dir",
        "tmp",
        "--save_h5ad",
        "tmp/benchmark.h5ad",
        "--parameter_dir",
        "tmp",
        "--save_benchmark",
        "tmp/benchmark.csv",
    ]


if __name__ == "__main__":
    default_parameter_test()  # use default parameter test case
    try:
        main()
    except Exception as e:
        import traceback

        print("Error", e)
        traceback.print_exc()
