import anndata as ad
import numpy as np
import scvelo as scv

try:
    # for docker
    from method_decorator import method_info
    from preprocess_pipeline import preprocess_pipeline
except ImportError:
    # for completed cafe environment
    from cafe.method.function.method_decorator import method_info
    from cafe.method.function.preprocess_pipeline import preprocess_pipeline


@method_info(
    name="veloae",
    version="0.0.1",
    description="VeloAE: Representation learning of RNA velocity reveals robust cell transitions",
    wrapper_type="velocity",
    doi="10.1073/pnas.2105859118",
    github_url="https://github.com/qiaochen/VeloAE",
    use_gpu=True,
    cpu_parallelization=True,
    available=True,
)
def veloae(
    adata: ad.AnnData,
    repreprocess: bool = True,
    repreprocess_kwargs: dict = {},
    veloae_args: dict = {},
):
    """VeloAE: Representation learning of RNA velocity reveals robust cell transitions"""
    # ref: https://github.com/qiaochen/VeloAE/blob/main/notebooks/pancreas/model-pancreas-gat.ipynb
    import torch
    from veloproj import (
        estimate_ld_velocity,
        fit_model,
        get_parser,
        init_model,
        new_adata,
    )

    # 1. preprocess
    if repreprocess:
        preprocess_pipeline(adata, style="scvelo", **repreprocess_kwargs)

    args = [
        "--lr",
        "1e-6",
        "--n-epochs",
        "10",
        "--g-rep-dim",
        "100",
        "--k-dim",
        "100",
        "--model-name",
        "pancreas_scv_model.cpt",
        "--exp-name",
        "CohAE_pancreas_scv",
        "--device",
        "cuda:0",
        "--gumbsoft_tau",
        "1",
        "--nb_g_src",
        "X",
        "--ld_nb_g_src",
        "X",
        "--n_raw_gene",
        "2000",
        "--n_conn_nb",
        "30",
        "--n_nb_newadata",
        "30",
        "--aux_weight",
        "1",
        "--fit_offset_train",
        "false",
        "--fit_offset_pred",
        "true",
        "--use_offset_pred",
        "false",
        "--gnn_layer",
        "GAT",
        "--vis-key",
        "X_umap",
        "--vis_type_col",
        "clusters",
        "--scv_n_jobs",
        "10",
    ]
    parser = get_parser()
    args = parser.parse_args(args=args)
    for k, v in veloae_args.items():
        if args.hasattr(k):
            setattr(args, k, v)
        else:
            print(f"Warning: args has no attribute {k}, please check the parameter name")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. execute scvelo stochastic model result
    # use scvelo stochastic model result
    scv.tl.velocity(adata, vkey="stc_velocity", mode="stochastic")
    # extract tensors
    spliced = adata.layers["Ms"]
    unspliced = adata.layers["Mu"]
    tensor_s = torch.FloatTensor(spliced).to(device)
    tensor_u = torch.FloatTensor(unspliced).to(device)
    tensor_x = torch.FloatTensor(adata.X.toarray()).to(device)
    tensor_v = torch.FloatTensor(adata.layers["stc_velocity"]).to(device)
    inputs = [tensor_s, tensor_u]
    xyids = [0, 1]
    if args.use_x:
        inputs.append(tensor_x)
    # model initialization
    model = init_model(adata, args, device)
    # model training
    model = fit_model(args, adata, model, inputs, tensor_v, xyids, device)
    # model inference to get high dimensional veocity
    model.eval()
    with torch.no_grad():
        x = model.encoder(tensor_x)
        s = model.encoder(tensor_s)
        u = model.encoder(tensor_u)

        v = (
            estimate_ld_velocity(
                s, u, device=device, perc=[5, 95], norm=args.use_norm, fit_offset=args.fit_offset_pred, use_offset=not args.use_offset_pred
            )
            .cpu()
            .numpy()
        )
        x = x.cpu().numpy()
        s = s.cpu().numpy()
        u = u.cpu().numpy()
        # project velocity to low-dim space
    adata = new_adata(adata, x, s, u, v, g_basis=args.ld_nb_g_src, n_nb_newadata=args.n_nb_newadata)
    scv.tl.velocity_graph(adata, vkey="new_velocity", n_jobs=args.scv_n_jobs)

    # 3,4. extract and save results
    trajectory_dict = {
        "wrapper_type": "velocity",
        "X": adata.X,
        "velocity": adata.layers["new_velocity"],
        "velocity_graph": adata.uns["new_velocity_graph"],
        "velocity_graph_neg": adata.uns["new_velocity_graph_neg"],
        "neighbors": {"distances": adata.obsp["distances"], "connectivities": adata.obsp["connectivities"]},
        "obs_index": adata.obs.index,
        "var_index": adata.var.index,  # here the var_index is the latent dimension, can't apply to now
    }

    return trajectory_dict
