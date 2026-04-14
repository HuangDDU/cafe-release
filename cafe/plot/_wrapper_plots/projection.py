import networkx as nx
import pandas as pd

from ..plot_trajectory import plot_trajectory

DEFAULT_MODE = "embedding"


def plot_embedding(fadata, model_name: str = None, basis: str = None):
    raw_wrapper_dict = fadata.get_raw_wrapper_dict(model_name)
    mw = fadata.get_milestone_wrapper(model_name)

    # milestone_network = raw_wrapper_dict["milestone_network"]
    X_emb = raw_wrapper_dict["X_emb"]
    # milestone_emb = raw_wrapper_dict["milestone_emb"]

    # plot trajectory
    axes = plot_trajectory(fadata, curve=False, show_milestone_labels=True, basis=basis)
    ax = axes.flatten()[0]

    # create edge dataframe

    trajectory_embedding = fadata.get_trajectory_embedding(model_name=model_name, basis=basis)
    milestone_positions = (
        trajectory_embedding["milestone_positions"].groupby("milestone_id").first()[["comp_1", "comp_2"]]
    )  # filter unique milestone pos

    def mix_emb(mpg, emb_df=milestone_positions):
        # mix related milestone emb to get position for a cell
        mpg_emb = emb_df.loc[mpg["milestone_id"]]
        return mpg_emb.apply(lambda emb_dim: (emb_dim.array * mpg["percentage"].array)).sum()

    raw_cell_emb_df = X_emb  # input X_emb
    projected_cell_emb_df = mw.milestone_percentages.groupby("cell_id").apply(lambda mpg: mix_emb(mpg))
    projected_cell_emb_df = projected_cell_emb_df.loc[raw_cell_emb_df.index]  # ensure index consistency
    projected_cell_emb_df.index = [f"projected_{i}" for i in projected_cell_emb_df.index]
    edge_df = pd.DataFrame(data=[raw_cell_emb_df.index.tolist(), projected_cell_emb_df.index.tolist()], index=["source", "target"]).T
    G = nx.from_pandas_edgelist(edge_df, create_using=nx.Graph)
    # add node position
    pos = {}
    pos.update(raw_cell_emb_df.apply(list, axis=1).to_dict())
    pos.update(projected_cell_emb_df.apply(list, axis=1).to_dict())

    # plot edges
    nx.draw_networkx_edges(G, pos=pos, ax=ax, alpha=0.5, edge_color="gray", width=0.5)

    return ax
