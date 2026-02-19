import pandas as pd
import plotly.express as px

from .. import logger
from ..data.fate_anndata import FateAnnData


def plot_embedding_plotly(adata, color=None, basis=None, obs_attribute_list=[]):
    # use plotly to plot embedding for interactive visualization with cell attribute hover info

    if isinstance(adata, FateAnnData):
        # use prior information if adata is FateAnnData
        if color is None:
            color = adata.prior_information.get("cluster")
            logger.debug(f"extract '{color}' from prior infomation as parameter 'color' ")
        if basis is None:
            basis = adata.prior_information.get("basis")
            logger.debug(f"extract '{basis}' from prior infomation as parameter 'basis' ")
    else:
        color = color if color is not None else "clusters"
        basis = basis if basis is not None else "X_umap"
        if color not in adata.obs.columns:
            logger.error(f"color '{color}' not in adata.obs columns")
        if basis not in adata.obsm.keys():
            logger.error(f"basis '{basis}' not in adata.obsm keys")

    # extract embedding coordinates
    x_name = f"{basis}1"
    y_name = f"{basis}2"
    emb_df = pd.DataFrame(data=adata.obsm[basis][:, :2], columns=[x_name, y_name], index=adata.obs_names)

    # remove existing x,y column names in obs
    obs = adata.obs.copy()
    for col in [x_name, y_name]:
        if col in obs.columns:
            obs = obs.drop(columns=[col])

    emb_df = emb_df.join(obs)

    # plot interactive UMAP with plotly
    fig = px.scatter(
        emb_df,
        x=x_name,
        y=y_name,
        color=color,  # color by cluster (replace with your obs column name)
        hover_name=adata.obs_names,  # main hover title is cell name
        hover_data={col: True for col in obs_attribute_list},  # show additional info
        width=800,
        height=600,
    )
    # show the figure
    fig.show()
