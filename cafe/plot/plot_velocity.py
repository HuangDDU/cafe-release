from ..util import temporary_obsm_key
from .util import save_fig


def plot_velocity(
    fadata,
    cluster=None,
    basis=None,
    model_name=None,
    style="scvelo",
    mode="stream",
    save: bool | str = None,
):
    cluster = fadata.check_cluster(cluster)
    basis = fadata.check_basis(basis)

    velocity_basis = f"velocity_{basis[2:]}"
    velocity_embedding = fadata.get_trajectory_pseudo_velocity(basis=basis, model_name=model_name)

    with temporary_obsm_key(fadata, velocity_basis, velocity_embedding):
        if style == "scvelo":
            import scvelo as scv

            if mode == "stream":
                ax = scv.pl.velocity_embedding_stream(fadata, color=cluster, basis=basis[2:], show=False)
            elif mode == "grid":
                ax = scv.pl.velocity_embedding_grid(fadata, color=cluster, basis=basis[2:], show=False)
            else:
                ax = scv.pl.velocity_embedding(fadata, color=cluster, basis=basis[2:], show=False)
        else:
            # TODO: dynamo style
            import dynamo as dyn

            dyn.pl.streamline_plot(fadata, color=cluster, basis=basis[2:])
    # TODO: save
    save_fig(save, default_filename=f".cafe/{fadata.id}/img/velocity_{model_name}.png", ax=ax)
