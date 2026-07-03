from ...util import temporary_obsm_key

DEFAULT_MODE = "embedding"  # embedding, grid, stream


# TODO: use ..plot_velocity
def plot_velocity(fadata, basis=None, model_name: str = None, style="scvelo", mode="stream"):
    if basis is None:
        basis = fadata.prior_information.get("basis")
    velocity_basis = f"velocity_{basis[2:]}"
    velocity_embedding = fadata.get_raw_wrapper_dict(model_name).get(velocity_basis)

    with temporary_obsm_key(fadata, velocity_basis, velocity_embedding):
        if style == "scvelo":
            import scvelo as scv

            if mode == "stream":
                ax = scv.pl.velocity_embedding_stream(fadata, basis=basis[2:], show=False)
            elif mode == "grid":
                ax = scv.pl.velocity_embedding_grid(fadata, basis=basis[2:], show=False)
            else:
                ax = scv.pl.velocity_embedding(fadata, basis=basis[2:], show=False)
        else:
            # TODO: dynamo style
            import dynamo as dyn

            ax = dyn.pl.streamline_plot(fadata, basis=basis[2:], save_show_or_return="return")

    return ax


def plot_embedding(fadata, model_name: str = None, basis: str = None, style="scvelo"):
    plot_velocity(fadata, basis=basis, model_name=model_name, style=style, mode="embedding")


def plot_stream(fadata, model_name: str = None, basis: str = None, style="scvelo"):
    plot_velocity(fadata, basis=basis, model_name=model_name, style=style, mode="stream")


def plot_grid(fadata, model_name: str = None, basis: str = None, style="scvelo"):
    plot_velocity(fadata, basis=basis, model_name=model_name, style=style, mode="grid")
