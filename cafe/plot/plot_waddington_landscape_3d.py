import os

import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import pandas as pd

# import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

# from IPython.display import display
from PIL import Image
from plotly.subplots import make_subplots
from scipy.interpolate import griddata

import cafe

from .._logging import logger


def merge_png_to_gif(folder, gif_name, duration=500):
    """
    Merge all PNG files in a folder into a GIF.

    Parameters
    ----------
    folder : str
        Folder containing PNG files.
    gif_name : str
        Output GIF file path.
    duration : int, default=500
        Frame duration in milliseconds.
    """
    png_files = [f for f in os.listdir(folder) if f.lower().endswith(".png")]
    png_files.sort()  # Sort by file name
    images = [Image.open(os.path.join(folder, f)) for f in png_files]
    if not images:
        print("png files are not found!")
        return
    images[0].save(gif_name, save_all=True, append_images=images[1:], duration=duration, loop=0)
    print(f"Saved GIF: {gif_name}")


def smooth_pseudotime_with_emb(pseudotime, embedding, start_idx, distance_weight=1):
    logger.debug("Smoothing pseudotime with embedding distribution")
    start_coord = embedding[start_idx]
    embedding_pseudotime = np.linalg.norm(embedding - start_coord, axis=1)
    embedding_pseudotime = (embedding_pseudotime - embedding_pseudotime.min()) / (embedding_pseudotime.max() - embedding_pseudotime.min())
    pseudotime = (pseudotime - pseudotime.min()) / (pseudotime.max() - pseudotime.min())
    pseudotime_smooth = (1 - distance_weight) * pseudotime + distance_weight * embedding_pseudotime
    return pseudotime_smooth


def plot_waddington_landscape_3d(
    fadata: cafe.data.FateAnnData,
    z_mode="pseudotime",  # pseudotime (continuous) or layer (discrete)
    groupby=None,
    pseudotime_key=None,
    pseudotime_model="cytotrace2",
    smooth_pseudotime=False,
    smooth_pseudotime_kwargs={},
    trajectory_model="ref",
    show_trajectory=True,
    start_milestone=None,
    auto_rescale_z=True,
    target_z_to_xy_ratio=0.45,
    z_scale=None,
    cols=3,
    camera_eye={"x": 1.5, "y": 1.5, "z": 0.5},
    figsize=(1000, 1000),
    cell_size=1,
    milestone_size=3,
    milestone_text_size=10,
    trajectory_width=5,
    landscape_opacity=0.5,
    arrow_size=1,
    impute_method="nearest",
    marker_kwargs={},
    png_width=None,
    png_height=None,
    save_path=None,
    save_png_dir=None,
    save_gif=True,
):
    """
    Plot an interactive 3D Waddington landscape with Plotly.

    Parameters
    ----------
    fadata : cafe.data.FateAnnData
        Input object containing cell annotations and trajectory information.
    z_mode : {"pseudotime", "layer"}, default="pseudotime"
        Strategy for constructing the Z axis.
        - "pseudotime": use continuous pseudotime values.
        - "layer": use milestone graph distance from `start_milestone`.
    groupby : str or None, default=None
        Column in `fadata.obs` used to split data into subplot panels.
    pseudotime_key : str or None, default=None
        Column name in `fadata.obs` for pseudotime values.
        If None, pseudotime is computed from `pseudotime_model`.
    pseudotime_model : str, default="cytotrace2"
        Model name used when computing pseudotime automatically.
    smooth_pseudotime : bool, default=False
        Whether to smooth pseudotime using embedding-based distances.
    smooth_pseudotime_kwargs : dict, default={}
        Additional keyword arguments for `smooth_pseudotime_with_emb()`.
    trajectory_model : str, default="ref"
        Model name used to retrieve trajectory embedding information.
    show_trajectory : bool, default=True
        Whether to overlay trajectory lines, arrows, and milestone points.
    start_milestone : str or None, default=None
        Starting milestone used when `z_mode="layer"`.
    auto_rescale_z : bool, default=True
        Whether to automatically rescale Z values so height is visually
        distinguishable relative to embedding span.
    target_z_to_xy_ratio : float, default=0.45
        Target ratio of Z span to XY span when `auto_rescale_z=True` and
        `z_scale` is None.
    z_scale : float or None, default=None
        Manual multiplier for Z values. If provided, overrides auto scaling.
    cols : int, default=3
        Number of columns in subplot layout.
    camera_eye : dict, default={"x": 1.5, "y": 1.5, "z": 0.5}
        Plotly camera eye position for 3D scenes.
    figsize : tuple[int, int], default=(1000, 1000)
        Figure size in pixels as `(width, height)`.
    cell_size : float, default=1
        Marker size for cells.
    milestone_size : float, default=3
        Marker size for milestone points.
    milestone_text_size : float, default=10
        Font size for milestone label text (e.g., A/B/C) shown in 3D plot.
    trajectory_width : float, default=5
        Line width for trajectory segments.
    landscape_opacity : float, default=0.5
        Opacity of the interpolated landscape surface.
    arrow_size : float, default=1
        Size scale for trajectory direction arrows.
    impute_method : {"linear", "nearest", "cubic"}, default="nearest"
        Interpolation method passed to `scipy.interpolate.griddata`.
    marker_kwargs : dict, default={}
        Extra marker settings currently used for legend marker size.
    png_width : int or None, default=None
        Width used for per-group PNG export; falls back to `figsize[0]`.
    png_height : int or None, default=None
        Height used for per-group PNG export; falls back to `figsize[1]`.
    save_path : str or None, default=None
        Output path for exported interactive HTML.
    save_png_dir : str or None, default=None
        If provided, save one PNG image per group into this folder.
    save_gif : bool, default=True
        If True and `save_png_dir` is provided, merge group PNGs into a GIF.

    Returns
    -------
    plotly.graph_objects.Figure
        The generated Plotly figure.
    """

    # --- 1. Prepare data ---
    cluster_key = fadata.prior_information["cluster"]
    basis = fadata.prior_information["basis"]

    if pseudotime_key is None:
        # Try to derive pseudotime_key from model name
        print(f"Loading pseudotime model: {pseudotime_model}...")
        pseudotime_key = f"{pseudotime_model}_pseudotime"
        fadata.load_trajectory_dict(pseudotime_model)
        fadata.obs[pseudotime_key] = fadata.get_trajectory_pseudotime(pseudotime_model)
    else:
        print(f"Using provided pseudotime_key '{pseudotime_key}' in fadata.obs")

    # Get embedding coordinates (X, Y)
    X_emb = fadata.obsm[basis]
    x_all = X_emb[:, 0]
    y_all = X_emb[:, 1]

    # Build Z values.
    # Waddington landscape is typically interpreted from high to low potential.
    if z_mode == "pseudotime":
        # pseudotime (continuous)
        pseudotime = fadata.obs[pseudotime_key].values
        # Optional pseudotime smoothing with embedding geometry
        if smooth_pseudotime:
            pseudotime = smooth_pseudotime_with_emb(
                pseudotime=pseudotime,
                embedding=X_emb,
                start_idx=fadata.obs.index.get_loc(fadata.prior_information["start_cell"]),
                **smooth_pseudotime_kwargs,
            )
        z_all = -pseudotime
    elif z_mode == "layer":
        # layer (discrete)
        # Build milestone graph and compute distances
        mw = fadata.get_milestone_wrapper(trajectory_model)
        milestone_network = mw.milestone_network
        is_directed = milestone_network["directed"].any()
        G = nx.from_pandas_edgelist(
            milestone_network,
            source="from",
            target="to",
            edge_attr=["length"],
            create_using=nx.DiGraph if is_directed else nx.Graph,
        )
        m_spl_dict = nx.shortest_path_length(G, source=start_milestone, weight="length")
        z_all = -fadata.obs[cluster_key].apply(lambda x: m_spl_dict.get(x, 0)).astype("float")
    else:
        raise ValueError("z_mode must be 'pseudotime' or 'layer'")

    # Rescale Z for better visual contrast against embedding span.
    # This preserves ordering/shape while expanding vertical discrimination.
    if z_scale is not None and z_scale <= 0:
        raise ValueError("z_scale should be positive when provided.")
    if target_z_to_xy_ratio <= 0:
        raise ValueError("target_z_to_xy_ratio should be positive.")

    x_span = float(np.nanmax(x_all) - np.nanmin(x_all)) if len(x_all) > 0 else 0.0
    y_span = float(np.nanmax(y_all) - np.nanmin(y_all)) if len(y_all) > 0 else 0.0
    xy_span = max(x_span, y_span, 1e-8)
    z_min = float(np.nanmin(z_all)) if len(z_all) > 0 else 0.0
    z_max = float(np.nanmax(z_all)) if len(z_all) > 0 else 0.0
    z_span = max(z_max - z_min, 1e-8)

    if z_scale is None and auto_rescale_z:
        z_scale_eff = (xy_span * target_z_to_xy_ratio) / z_span
    elif z_scale is None:
        z_scale_eff = 1.0
    else:
        z_scale_eff = float(z_scale)

    z_center = 0.5 * (z_min + z_max)
    z_all = (z_all - z_center) * z_scale_eff + z_center

    # Get cell type labels and colors
    if cluster_key in fadata.obs:
        cell_types_all = fadata.obs[cluster_key].astype(str)
        unique_types = np.sort(cell_types_all.unique())
        # Get colors and convert to Hex for Plotly
        if f"{cluster_key}_colors" in fadata.uns:
            categories = fadata.obs[cluster_key].cat.categories
            color_list = fadata.uns[f"{cluster_key}_colors"]
            # Ensure color format is Hex
            color_list = [mcolors.to_hex(c) for c in color_list]
            color_map = dict(zip(categories, color_list))
        else:
            palette = sns.color_palette("tab20", len(unique_types))
            color_list = [mcolors.to_hex(c) for c in palette]
            color_map = dict(zip(unique_types, color_list))
    else:
        cell_types_all = pd.Series(["Cell"] * len(x_all))
        unique_types = ["Cell"]
        color_map = {"Cell": "#1f77b4"}

    # --- 2. Fit Waddington landscape (global interpolation surface) ---
    # Use global data so all subplots share a consistent surface
    grid_x, grid_y = np.mgrid[min(x_all) : max(x_all) : 100j, min(y_all) : max(y_all) : 100j]
    # fill_value=max(z): set outside-convex-hull area to high potential
    grid_z = griddata((x_all, y_all), z_all, (grid_x, grid_y), method=impute_method, fill_value=np.max(z_all))

    # --- 3. Prepare group data ---
    if groupby is not None:
        group_series = fadata.obs[groupby].reset_index(drop=True)
        groups = group_series.cat.categories.to_list()
        n_groups = len(groups)
        # Calculate subplot rows/cols
        cols = min(cols, n_groups)
        rows = (n_groups + cols - 1) // cols
        subplot_titles = [str(g) for g in groups]
    else:
        groups = ["All"]
        rows = 1
        cols = 1
        subplot_titles = ["Waddington Landscape"]

    # --- 4. Create Plotly figure ---
    fig = make_subplots(
        rows=rows,
        cols=cols,
        specs=[[{"type": "surface"}] * cols] * rows,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.01,
        vertical_spacing=0.01,  # Minimal spacing
    )

    # Common traces: landscape, trajectory (waypoint + arrow), milestones
    common_trace_surface = go.Surface(
        z=grid_z, x=grid_x, y=grid_y, colorscale="Viridis", opacity=landscape_opacity, showscale=False, name="Landscape", hoverinfo="skip"
    )
    common_trace_list = [common_trace_surface]
    if show_trajectory:
        # Retrieve trajectory embedding
        traj_emb = fadata.get_trajectory_embedding(model_name=trajectory_model)
        if traj_emb is None:
            cafe.plot.plot_trajectory(fadata, model_name=trajectory_model, sc_pl_embedding_kwargs={"show": False})
            traj_emb = fadata.get_trajectory_embedding(model_name=trajectory_model)

        # Milestone positions (for points)
        milestone_positions = traj_emb["milestone_positions"].copy()
        milestone_positions = milestone_positions.groupby("milestone_id").apply(lambda x: x.iloc[0]).reset_index(drop=True)

        # Waypoints (for trajectory curves)
        wp_segments = traj_emb["wp_segments"].copy()

        # Compute Z coordinates for trajectory/milestones from global surface
        # (1) milestone Z
        mx = milestone_positions.iloc[:, 0].values
        my = milestone_positions.iloc[:, 1].values
        mz = griddata((x_all, y_all), z_all, (mx, my), method=impute_method)
        mask_nan = np.isnan(mz)
        if mask_nan.any():
            mz[mask_nan] = griddata((x_all, y_all), z_all, (mx[mask_nan], my[mask_nan]), method="nearest")
        milestone_positions["z"] = mz
        # (2) Waypoints Z
        wx = wp_segments["comp_1"].values
        wy = wp_segments["comp_2"].values
        wz = griddata((x_all, y_all), z_all, (wx, wy), method=impute_method)
        mask_nan_w = np.isnan(wz)
        if mask_nan_w.any():
            wz[mask_nan_w] = griddata((x_all, y_all), z_all, (wx[mask_nan_w], wy[mask_nan_w]), method="nearest")
        wp_segments["z"] = wz

        # Build trajectory traces
        common_trace_trajectory_list = []
        for g in wp_segments["group"].unique():
            wp_g = wp_segments[wp_segments["group"] == g]
            common_trace_trajectory_list.append(
                go.Scatter3d(
                    x=wp_g["comp_1"],
                    y=wp_g["comp_2"],
                    z=wp_g["z"],
                    mode="lines",
                    line=dict(color="black", width=trajectory_width),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )  # waypoint
            arrow_points = wp_g[wp_g["arrow"] == True]  # noqa: E712
            if not arrow_points.empty:
                for idx in arrow_points.index:
                    curr_pos = wp_segments.loc[idx]
                    next_idx = idx + 1
                    if next_idx in wp_g.index:
                        next_pos = wp_segments.loc[next_idx]
                        u = next_pos["comp_1"] - curr_pos["comp_1"]
                        v = next_pos["comp_2"] - curr_pos["comp_2"]
                        w = next_pos["z"] - curr_pos["z"]

                        common_trace_trajectory_list.append(
                            go.Cone(
                                x=[curr_pos["comp_1"]],
                                y=[curr_pos["comp_2"]],
                                z=[curr_pos["z"]],
                                u=[u],
                                v=[v],
                                w=[w],
                                sizemode="absolute",
                                sizeref=arrow_size,
                                anchor="tail",
                                colorscale=[[0, "black"], [1, "black"]],
                                showscale=False,
                                name="Arrow",
                                showlegend=False,
                            )
                        )  # arrow
        common_trace_trajectory_list.append(
            go.Scatter3d(
                x=milestone_positions.iloc[:, 0],
                y=milestone_positions.iloc[:, 1],
                z=milestone_positions["z"],
                mode="markers+text" if milestone_text_size > 0 else "markers",
                marker=dict(size=milestone_size, color="black"),
                text=milestone_positions["milestone_id"],
                textposition="top center",
                # textfont=dict(size=milestone_text_size),
                name="Milestones",
                showlegend=False,
            )
        )  # milestone
        common_trace_list += common_trace_trajectory_list  # common traces

    for i, group_name in enumerate(groups):
        row = i // cols + 1
        col = i % cols + 1

        # Filter data for current group
        if groupby is not None:
            mask = (group_series == group_name).values
        else:
            mask = np.ones(len(x_all), dtype=bool)
        x_g = x_all[mask]
        y_g = y_all[mask]
        z_g = z_all[mask]
        cell_types_g = cell_types_all[mask]

        # Build cell traces
        cell_trace_list = []
        for c_type in np.unique(cell_types_g):
            type_mask = cell_types_g == c_type
            cell_trace_list.append(
                go.Scatter3d(
                    x=x_g[type_mask],
                    y=y_g[type_mask],
                    z=z_g[type_mask],
                    mode="markers",
                    marker=dict(size=cell_size, color=color_map.get(c_type, "#333333"), opacity=0.8),
                    name=c_type,
                    showlegend=False,
                    legendgroup=c_type,
                ),
            )

        # Draw traces
        # Bottom layer: common traces (surface + optional trajectory/milestones)
        for common_trace in common_trace_list:
            fig.add_trace(common_trace, row=row, col=col)
        # Top layer: cells
        for cell_trace in cell_trace_list:
            fig.add_trace(cell_trace, row=row, col=col)

        if save_png_dir:
            # Save each group subplot as a standalone PNG
            import plotly.io as pio

            # width, height = figsize[0] // cols, figsize[1] // rows # too small
            # width, height = figsize[0], figsize[1]
            width = png_width if png_width else figsize[0]
            height = png_height if png_height else figsize[1]
            single_fig = go.Figure()
            for common_trace in common_trace_list:
                single_fig.add_trace(common_trace)
            for cell_trace in cell_trace_list:
                single_fig.add_trace(cell_trace)
            single_fig.update_layout(
                title_text=group_name,
                margin=dict(l=0, r=0, b=0, t=40),
                width=width,
                height=height,
                scene=dict(
                    xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode="cube", camera=dict(eye=camera_eye)
                ),
            )
            png_path = f"{save_png_dir}/{i+1}_{group_name}.png"  # Keep deterministic ordering
            single_fig.write_image(png_path, scale=2)
            pio.write_image(single_fig, png_path, width=width, height=height)  # plotly-6.0.1 with kaleido-0.2.0
            print(f"Saved PNG for group '{group_name}' to {png_path}")

    # Merge PNGs into a GIF
    if groupby is not None and save_png_dir and save_gif:
        merge_png_to_gif(save_png_dir, os.path.join(save_png_dir, "merged.gif"))

    # Draw a global cell legend
    marker_size = marker_kwargs.get("size", cell_size * 5)
    for c_type in fadata.obs[cluster_key].cat.categories:
        fig.add_trace(
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode="markers",
                marker=dict(size=marker_size, color=color_map.get(c_type, "#333333"), opacity=1),
                name=c_type,
                showlegend=True,
                legendgroup=c_type,
            )
        )

    # --- 6. Configure layout ---
    # Set final global layout and legend
    layout_dict = dict(
        title="3D Waddington Landscape",
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(yanchor="top", xanchor="left", font=dict(size=20), itemsizing="constant"),
        width=figsize[0],
        height=figsize[1],
        # scene_camera=camera_eye
    )

    # Configure axis properties for each scene
    for i in range(1, rows * cols + 1):
        scene_name = f"scene{i}" if i > 1 else "scene"
        axis_parameter_dict = {
            "backgroundcolor": "rgba(0,0,0,0)",
            "gridcolor": "lightgrey",
            "showbackground": False,
            "showgrid": False,
            "zeroline": False,
            "showticklabels": False,
            "title": "",
        }
        layout_dict[scene_name] = dict(
            xaxis=dict(**axis_parameter_dict),
            yaxis=dict(**axis_parameter_dict),
            zaxis=dict(**axis_parameter_dict),
            # xaxis=dict(title=f"{basis}_1", **axis_parameter_dict),
            # yaxis=dict(title=f"{basis}_2", **axis_parameter_dict),
            # zaxis=dict(title="Potential",**axis_parameter_dict),
            aspectmode="cube",
        )
        fig.update_layout({f"{scene_name}_camera": dict(eye=camera_eye)})

    fig.update_layout(**layout_dict)

    # --- 7. Save and return ---
    if save_path:
        fig.write_html(save_path)
        print(f"Figure saved to {save_path}")
    # Display in notebook when explicitly needed
    # display(fig)
    return fig
