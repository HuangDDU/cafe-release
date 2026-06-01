import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import logger


def visualize(benchmark_df, save="benchmark_heatmap.pdf", add_method_meta=False, figsize=(32, 24)):
    # Visualize benchmark results using funkyheatmappy.
    # adapt to pancreas benchmark results, with method metadata and resource usage info.
    import funkyheatmappy

    benchmark_df["id"] = benchmark_df.index
    plt.figure(figsize=figsize)

    column_list = [
        ["id", "group", "name", "geom", "options", "palette"],
        ["id", np.nan, "", "text", {"ha": 0, "width": 4}, np.nan],
        # method meta data
        # ["type", np.nan, "Wrapper Type", "text", {"ha": 0, "width": 4}, np.nan],
        # ["prior_level", np.nan, "Prior Level", "text", {"ha": 0, "width": 4}, np.nan],
    ]

    if "overall" in benchmark_df.columns:
        column_list.append(["overall", "overall", "Overall", "bar", {"width": 4, "legend": False}, "overall_palette"])

    # Accuracy metrics
    id2name = {
        "pseudotime_correlation": "Pseudotime Correlation",
        "velocity_cbdir": "Velocity CBDir",
        "velocity_icvcoh": "Velocity ICVCoh",
        "edge_flip": "Edge Flip",
        "him": "HIM",
        # "isomorphic": "Isomorphic", # 0 for all rows may result in funkyheatmap error
        "F1_branches": "F1 Branches",
        "F1_milestones": "F1 Milestones",
        "correlation": "Correlation",
        "rf_mse": "RF MSE",
        "rf_nmse": "RF NMSE",
        "rf_rsq": "RF R^2",
        "lm_nmse": "LM NMSE",
        "lm_mse": "LM MSE",
        "lm_rsq": "LM R^2",
        "featureimp_cor": "Feature Imp Correlation",
        "featureimp_wcor": "Feature Imp Weighted Correlation",
    }

    metric_column_list = []
    for id, name in id2name.items():
        if id in benchmark_df.columns:
            metric_column_list.append([id, "metric", name, "funkyrect", {"size": 3}, "metric_palette"])
    column_list = column_list + metric_column_list

    # resource group
    if "time" in benchmark_df.columns:
        benchmark_df["time"] = -benchmark_df["time"]  # invert time for better color visualization (shorter time -> darker color)
        column_list.append(["time", "resource", "Time", "rect", "scaling", "resource_palette"])
        column_list.append(["time_text", "resource", "", "text", {"overlay": True, "size": 3, "scale": False}, "white6black4"])

    if "memory" in benchmark_df.columns:
        benchmark_df["memory"] = -benchmark_df["memory"]  # invert memory for better color visualization (less memory -> darker color)
        column_list.append(["memory", "resource", "Memory", "rect", "scaling", "resource_palette"])
        column_list.append(["memory_text", "resource", "", "text", {"overlay": True, "size": 3, "scale": False}, "white6black4"])

    column_info = pd.DataFrame(column_list[1:], columns=column_list[0])
    column_info.index = column_info["id"]

    column_groups = pd.DataFrame(
        columns=["Category", "group", "palette"],
        data=[["Overall", "overall", "overall_palette"], ["Metric", "metric", "metric_palette"], ["Resources", "resource", "resource_palette"]],
    )
    # palette definitions
    palettes = {
        "overall_palette": "Blues",
        "metric_palette": "Greens",
        "resource_palette": "YlOrBr",
    }

    row_info = None
    row_groups = None

    if add_method_meta:
        if isinstance(add_method_meta, pd.DataFrame):
            method_meta_df = add_method_meta.copy()
        else:
            from .method_meta import get_method_meta

            method_meta_df = get_method_meta(regenerate=False)

        # Merge metadata columns on method id while preserving benchmark row order.
        benchmark_df = benchmark_df.merge(method_meta_df, left_index=True, right_index=True, how="left", suffixes=("", "_meta"))

        # Use method display name from metadata if available.
        display_name_col = None
        for candidate in ["Name", "name", "Name_meta", "name_meta"]:
            if candidate in benchmark_df.columns:
                display_name_col = candidate
                break
        if display_name_col is not None:
            benchmark_df["id"] = benchmark_df[display_name_col].fillna(benchmark_df.index.astype(str).to_series()).astype(str)

        if "stars" in benchmark_df.columns:
            benchmark_df["stars"] = benchmark_df["stars"].astype("Int64").astype(str).replace("<NA>", "")
            column_list.append(["stars", "meta", "Star", "text", {}, "white6black4"])
        if "citations" in benchmark_df.columns:
            benchmark_df["citations"] = benchmark_df["citations"].astype("Int64").astype(str).replace("<NA>", "")
            column_list.append(["citations", "meta", "Citations", "text", {}, "white6black4"])
        if "prior_level" in benchmark_df.columns:
            column_list.append(["prior_level", "meta", "Priors required", "text", {"size": 0.3, "scale": True}, "meta_palette"])

        if any(x[1] == "meta" for x in column_list if len(x) > 1):
            column_groups.loc[len(column_groups)] = ["Meta", "meta", "meta_palette"]
            palettes["meta_palette"] = "Greys"

        group_col = None
        for candidate in ["type", "wrapper_type", "type_meta", "wrapper_type_meta"]:
            if candidate in benchmark_df.columns:
                group_col = candidate
                break

        if group_col is not None:
            group_values = benchmark_df[group_col]
            if pd.api.types.is_categorical_dtype(group_values):
                group_list = [g for g in group_values.cat.categories if g in set(group_values.dropna())]
            else:
                group_list = [g for g in pd.unique(group_values) if pd.notna(g)]

            row_info = benchmark_df[[group_col]].rename(columns={group_col: "group"})
            row_info["id"] = benchmark_df["id"].values
            if len(group_list) > 0:
                row_info["group"] = pd.Categorical(row_info["group"], categories=group_list, ordered=True)
                if "overall" in benchmark_df.columns:
                    benchmark_df["overall"] = pd.to_numeric(benchmark_df["overall"], errors="coerce")
                    # sort all
                    sort_df = pd.DataFrame(
                        {
                            "group": row_info["group"],
                            "overall": benchmark_df["overall"],
                        },
                        index=benchmark_df.index,
                    )
                    sort_df = sort_df.sort_values(["group", "overall"], ascending=[True, False], na_position="last")
                    benchmark_df = benchmark_df.loc[sort_df.index]
                    row_info = row_info.loc[sort_df.index]
                else:
                    row_info = row_info.sort_values("group")
                    benchmark_df = benchmark_df.loc[row_info.index]
                row_groups = pd.DataFrame({"group": group_list, "Group": group_list})
            row_info.index = row_info["id"]

    column_info = pd.DataFrame(column_list[1:], columns=column_list[0])
    column_info.index = column_info["id"]

    # Render prior requirement as image badges.
    if "prior_level" in column_info.index:
        prior_img_dir = f"{os.path.dirname(__file__)}/funkyheatmap/img/"
        if os.path.isdir(prior_img_dir):
            column_info.loc["prior_level", "geom"] = "image"
            column_info.loc["prior_level", "path"] = prior_img_dir
            column_info.loc["prior_level", "filetype"] = "png"

    funkyheatmappy.funky_heatmap(
        benchmark_df,
        column_info=column_info,
        column_groups=column_groups,
        row_info=row_info,
        row_groups=row_groups,
        palettes=palettes,
    )

    if save is not None:
        if isinstance(save, bool) and save:
            save = ".cafe/img/benchmark_funkyheatmap.png"  #
        os.makedirs(os.path.dirname(save), exist_ok=True) if os.path.dirname(save) else None
        plt.savefig(save, bbox_inches="tight")
        logger.debug(f"save benchmark plot to '{save}'")

    return benchmark_df


def show_trajectory_result_grid(fadata, model_name_list, save="benchmark_trajectory_grid.pdf"):
    import patchworklib as pw
    from IPython.display import Image, display
    from IPython.utils.capture import capture_output

    from ..plot import plot_graph, plot_trajectory, plot_wrapper

    sc_pl_embedding_kwargs = {"legend_loc": None, "title": ""}
    column_titles = ("Trajectory", "Graph", "Wrapper")
    row_label_width = 1.4
    cell_figsize = (3.2, 2.4)
    header_figsize = (cell_figsize[0], 0.45)

    def _extract_first_ax(obj):
        if obj is None:
            return None
        if hasattr(obj, "plot") and hasattr(obj, "figure"):
            return obj
        if isinstance(obj, np.ndarray):
            if obj.size == 0:
                return None
            return _extract_first_ax(obj.flat[0])
        if isinstance(obj, (list, tuple)):
            if len(obj) == 0:
                return None
            return _extract_first_ax(obj[0])
        return None

    def _fig_score(fig):
        if fig is None:
            return (-1, -1)

        axes = fig.axes
        n_axes = len(axes)
        artist_score = 0
        for ax in axes:
            artist_score += len(ax.lines)
            artist_score += len(ax.collections)
            artist_score += len(ax.patches)
            artist_score += len(ax.images)
            artist_score += len(ax.texts)
        return (n_axes, artist_score)

    def _pick_best_figure(fig_list):
        if len(fig_list) == 0:
            return None
        return max(fig_list, key=_fig_score)

    def _trim_white_border(img, white_threshold=245, pad=6):
        if img.size == 0:
            return img
        non_white = np.any(img < white_threshold, axis=2)
        coords = np.argwhere(non_white)
        if coords.size == 0:
            return img
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        y0 = max(0, y0 - pad)
        x0 = max(0, x0 - pad)
        y1 = min(img.shape[0], y1 + pad)
        x1 = min(img.shape[1], x1 + pad)
        return img[y0:y1, x0:x1]

    def _pad_to_aspect_ratio(img, target_ratio):
        h, w = img.shape[:2]
        if h == 0 or w == 0:
            return img
        current_ratio = w / h
        if abs(current_ratio - target_ratio) < 1e-3:
            return img

        if current_ratio > target_ratio:
            new_h = int(np.ceil(w / target_ratio))
            pad_total = max(0, new_h - h)
            pad_top = pad_total // 2
            pad_bottom = pad_total - pad_top
            return np.pad(img, ((pad_top, pad_bottom), (0, 0), (0, 0)), mode="constant", constant_values=255)

        new_w = int(np.ceil(h * target_ratio))
        pad_total = max(0, new_w - w)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return np.pad(img, ((0, 0), (pad_left, pad_right), (0, 0)), mode="constant", constant_values=255)

    def _fig_to_rgb_array(fig, target_ax=None):
        from matplotlib.legend import Legend

        for ax in list(fig.axes):
            # scanpy continuous plots may add colorbar axes; remove them for clean compact panels
            if ax.get_label() == "<colorbar>":
                fig.delaxes(ax)
                continue

            ax.set_title("")
            for child in list(ax.get_children()):
                if isinstance(child, Legend):
                    child.remove()
        try:
            fig.tight_layout(pad=0.1)
        except Exception:
            pass
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(h, w, 3)

        # If a specific axis is known (e.g., first subplot in a multi-panel figure),
        # crop to that axis region so the rendered panel size is consistent across methods.
        if target_ax is not None and target_ax in fig.axes:
            pos = target_ax.get_position()
            x0 = max(0, int(pos.x0 * w))
            x1 = min(w, int(pos.x1 * w))
            y0 = max(0, int((1 - pos.y1) * h))
            y1 = min(h, int((1 - pos.y0) * h))
            if x1 > x0 and y1 > y0:
                img = img[y0:y1, x0:x1]

        img = _trim_white_border(img)
        target_ratio = cell_figsize[0] / cell_figsize[1]
        img = _pad_to_aspect_ratio(img, target_ratio)
        return img

    def _make_image_brick(img, figsize=cell_figsize):
        br = pw.Brick(figsize=figsize)
        br.imshow(img)
        br.set_axis_off()
        return br

    def _make_text_brick(text, figsize=(1.0, 0.45), fontsize=11, weight="normal", ha="center"):
        br = pw.Brick(figsize=figsize)
        x = 0.02 if ha == "left" else 0.5
        br.text(x, 0.5, text, ha=ha, va="center", fontsize=fontsize, fontweight=weight)
        br.set_axis_off()
        return br

    def _resolve_wrapper_mode(model_name):
        raw_wrapper_dict = fadata.get_raw_wrapper_dict(model_name=model_name)
        wrapper_type = raw_wrapper_dict.get("wrapper_type")
        if wrapper_type == "velocity":
            return "stream"
        if wrapper_type in {"probability", "lineage"}:
            # use single-panel star view for probabilistic lineages (e.g. CellRank)
            return "star"
        return None

    def _render_call_to_image(plot_call):
        before_nums = set(plt.get_fignums())
        with capture_output():
            ret = plot_call()

        after_nums = set(plt.get_fignums())
        new_nums = [n for n in after_nums if n not in before_nums]
        new_figs = [plt.figure(n) for n in new_nums]

        target_ax = _extract_first_ax(ret)
        if target_ax is not None and hasattr(target_ax, "figure"):
            target_fig = target_ax.figure
        else:
            target_fig = _pick_best_figure(new_figs)
            target_ax = target_fig.axes[0] if (target_fig is not None and len(target_fig.axes) > 0) else None

        if target_fig is None:
            target_fig = plt.gcf()

        img = _fig_to_rgb_array(target_fig, target_ax=target_ax)
        for fig in new_figs:
            plt.close(fig)
        return img

    pw_rows = []

    for model_name in model_name_list:
        fadata.load_trajectory_dict(model_name)
        wrapper_mode = _resolve_wrapper_mode(model_name)

        img_traj = _render_call_to_image(
            lambda mn=model_name: plot_trajectory(
                fadata,
                model_name=mn,
                **sc_pl_embedding_kwargs,
            )
        )
        img_graph = _render_call_to_image(
            lambda mn=model_name: plot_graph(
                fadata,
                model_name=mn,
                **sc_pl_embedding_kwargs,
            )
        )
        if wrapper_mode == "star":
            try:
                img_wrapper = _render_call_to_image(
                    lambda mn=model_name, wm=wrapper_mode: plot_wrapper(
                        fadata,
                        model_name=mn,
                        mode=wm,
                    )
                )
            except Exception as e:
                logger.warning(f"wrapper mode 'star' failed for model '{model_name}', fallback to default mode: {e}")
                img_wrapper = _render_call_to_image(
                    lambda mn=model_name: plot_wrapper(
                        fadata,
                        model_name=mn,
                        mode=None,
                    )
                )
        else:
            img_wrapper = _render_call_to_image(
                lambda mn=model_name, wm=wrapper_mode: plot_wrapper(
                    fadata,
                    model_name=mn,
                    mode=wm,
                )
            )

        method_prefix = str(model_name)
        # wrapper type or method type info from method_meta_df if available
        # if isinstance(method_meta_df, pd.DataFrame) and model_name in method_meta_df.index:
        #     if "wrapper_type" in method_meta_df.columns and pd.notna(method_meta_df.loc[model_name, "wrapper_type"]):
        #         method_prefix = f"{model_name} ({method_meta_df.loc[model_name, 'wrapper_type']})"
        #     elif "type" in method_meta_df.columns and pd.notna(method_meta_df.loc[model_name, "type"]):
        #         method_prefix = f"{model_name} ({method_meta_df.loc[model_name, 'type']})"

        br_row_label = _make_text_brick(method_prefix, figsize=(row_label_width, cell_figsize[1]), fontsize=12, weight="bold", ha="left")
        br_traj = _make_image_brick(img_traj)
        br_graph = _make_image_brick(img_graph)
        br_wrapper = _make_image_brick(img_wrapper)
        pw_rows.append(br_row_label | br_traj | br_graph | br_wrapper)

    if len(pw_rows) == 0:
        raise ValueError("benchmark_df is empty, cannot build trajectory result grid.")

    header_pw = _make_text_brick("", figsize=(row_label_width, header_figsize[1]))
    for title in column_titles:
        header_pw = header_pw | _make_text_brick(title, figsize=header_figsize, fontsize=12, weight="bold")

    final_pw = header_pw
    for row in pw_rows:
        final_pw = final_pw / row

    if save is not None:
        if isinstance(save, bool) and save:
            save = ".cafe/img/benchmark_trajectory_grid.png"
        os.makedirs(os.path.dirname(save), exist_ok=True) if os.path.dirname(save) else None
        final_pw.savefig(save, dpi=300)
        logger.debug(f"save benchmark trajectory grid to '{save}'")
        try:
            display(Image(filename=save))
        except Exception:
            # If image display backend is unavailable, still return the patchwork object.
            pass
    else:
        try:
            display(final_pw)
        except Exception:
            pass

    return final_pw
