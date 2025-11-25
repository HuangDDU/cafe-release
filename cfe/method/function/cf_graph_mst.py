import anndata as ad
import networkx as nx

try:
    # for docker
    from method_decorator import method_info
    from preprocess_pipeline import preprocess_pipeline
except ImportError:
    # for completed cfe environment
    from cfe.method.function.method_decorator import method_info
    from cfe.method.function.preprocess_pipeline import preprocess_pipeline


@method_info(
    name="graph_mst",
    version="0.0.1",
    description="Graph MST: baseline for graph wrapper, creating a Minimum Spanning Tree (MST) on cluster centers.",
    wrapper_type="graph",
)
def graph_mst(
    adata: ad.AnnData,
    repreprocess: bool = True,
):
    # 1. preprocess
    if repreprocess:
        preprocess_pipeline(adata, style="scanpy", if_neighbors=True)  # ensure neighbors are computed

    # 2.execute method
    cell_id_list = adata.obs.index.tolist()
    G = nx.from_scipy_sparse_array(adata.obsp["distances"])  # construct graph from a sparse matrix
    cell_mst = nx.minimum_spanning_tree(G, weight="weight")  # construct the minimum spanning

    # 3. extract results
    cell_graph = nx.to_pandas_edgelist(cell_mst, source="from", target="to").rename(columns={"weight": "length"})
    cell_graph["from"] = cell_graph["from"].apply(lambda x: cell_id_list[x])
    cell_graph["to"] = cell_graph["to"].apply(lambda x: cell_id_list[x])
    # to_keep = pd.Series(data=True, index=cell_ids)

    # 4. save results
    trajectory_dict = {
        "wrapper_type": "graph",
        "cell_graph": cell_graph,
        "to_keep": None,  # keep all
    }
    return trajectory_dict
