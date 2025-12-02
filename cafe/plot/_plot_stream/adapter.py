"""
Stream plot adapter for FateAnndata with MilestoneWrapper

将 FateAnndata 的 MilestoneWrapper 转换为 phlower stream.py 所需的数据结构
并调用 stream.py 的绘图函数进行可视化

STREAM 绘图要求的数据结构：
1. adata.uns['stream_tree']: nx.Graph
   - 节点属性: 'label' (如 'S0', 'S1'), 'original' (原始名称元组)
   - 边属性: 'id', 'len', 'color', 'nodes'
2. adata.obs['branch_id']: 每个细胞所在分支的 tuple (from, to)
3. adata.obs['branch_id_alias']: 分支的别名
4. adata.obs['branch_lam']: 细胞在分支上的位置
5. adata.obs['branch_dist']: 细胞到分支的距离
6. adata.obs['<root>_pseudotime']: 从根节点出发的伪时间
"""

from typing import Dict, List, Tuple

import colorcet as cc
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from ..._logging import logger
from ...data import FateAnnData, MilestoneWrapper


class StreamPlotAdapter:
    """适配器类：将 FateAnndata + MilestoneWrapper 转换为 stream.py 兼容格式"""

    def __init__(self, fadata: FateAnnData, model_name: str = None):
        """
        初始化适配器

        Args:
            fadata: FateAnnData 对象
            model_name: 指定使用哪个模型的轨迹（如果为None则使用当前模型）
        """
        self.fadata = fadata
        self.model_name = model_name or fadata.model_name
        self.milestone_wrapper: MilestoneWrapper = fadata.get_milestone_wrapper(model_name=self.model_name)

    def build_stream_tree(self, root_milestone: str = None) -> nx.Graph:
        """
        从 MilestoneWrapper 的 milestone_network 构建 stream.py 兼容的 networkx.Graph

        STREAM 要求的节点属性：
        - 'label': 简化标签如 'S0', 'S1' 等
        - 'original': 原始名称（tuple 形式）

        边属性：
        - 'id': 边的标识符 tuple
        - 'len': 边的长度
        - 'color': 边的颜色
        - 'nodes': 边上的节点列表

        注意：stream.py 期望通过 'root' 节点名来访问根节点，其 'label' 属性返回实际根节点的标签。
        我们通过将实际根节点重命名为 'root' 来实现这一点。

        Args:
            root_milestone: 指定根节点的 milestone_id

        Returns:
            nx.Graph: stream.py 兼容的图
        """
        milestone_network = self.milestone_wrapper.milestone_network

        # STREAM 使用无向图
        G = nx.Graph()

        # 默认根节点
        if root_milestone is None:
            root_milestone = self.milestone_wrapper.id_list[0]

        # 为每个 milestone 创建节点，分配 'S0', 'S1' 等标签
        # 根节点使用 'root' 作为节点名，但 label 仍为 'S0'
        node_to_label = {}
        for i, milestone_id in enumerate(self.milestone_wrapper.id_list):
            label = f"S{i}"
            node_to_label[milestone_id] = label

            # 如果是根节点，使用 'root' 作为节点名
            if milestone_id == root_milestone:
                node_name = "root"
            else:
                node_name = milestone_id

            # 'original' 存储为 tuple
            G.add_node(node_name, label=label, original=(milestone_id,))

        # 添加边，将边中涉及根节点的部分改为 'root'
        palette = sns.color_palette(cc.glasbey, n_colors=len(milestone_network)).as_hex()
        for idx, row in milestone_network.iterrows():
            from_node = row["from"]
            to_node = row["to"]
            length = float(row.get("length", 1.0))

            # 如果节点是根节点，替换为 'root'
            from_name = "root" if from_node == root_milestone else from_node
            to_name = "root" if to_node == root_milestone else to_node

            G.add_edge(
                from_name,
                to_name,
                id=(from_name, to_name),
                len=length,
                color=palette[idx % len(palette)],
                nodes=[from_name, to_name],
            )

        # 保存原始 milestone_id 到节点名的映射
        self._milestone_to_node = {mid: ("root" if mid == root_milestone else mid) for mid in self.milestone_wrapper.id_list}
        self._root_milestone = root_milestone

        return G

    def _compute_node_positions_from_progressions(self, embedding_basis: str = "X_umap") -> Dict[str, np.ndarray]:
        """
        基于 progressions 和 embedding 计算每个 milestone 节点的位置

        对于每个 milestone，找出该 milestone 占比最高的细胞，
        然后计算这些细胞在 embedding 中的加权平均位置

        节点名会根据 _milestone_to_node 映射进行转换

        Returns:
            node_positions: {node_name: np.array([x, y])}
        """
        X_emb = self.fadata.obsm[embedding_basis]
        cell_ids = self.fadata.obs.index.tolist()
        cell_id_to_idx = {cid: i for i, cid in enumerate(cell_ids)}

        milestone_percentages = self.milestone_wrapper.milestone_percentages

        # 获取 milestone 到节点名的映射（如果存在）
        milestone_to_node = getattr(self, "_milestone_to_node", None)
        if milestone_to_node is None:
            milestone_to_node = {mid: mid for mid in self.milestone_wrapper.id_list}

        node_positions = {}

        for milestone_id in self.milestone_wrapper.id_list:
            # 获取该 milestone 的所有细胞及其百分比
            mp_subset = milestone_percentages[milestone_percentages["milestone_id"] == milestone_id]

            # 获取转换后的节点名
            node_name = milestone_to_node.get(milestone_id, milestone_id)

            if len(mp_subset) == 0:
                # 如果没有细胞，使用全局中心
                node_positions[node_name] = X_emb.mean(axis=0)
                continue

            # 计算加权平均位置
            weighted_pos = np.zeros(X_emb.shape[1])
            total_weight = 0.0

            for _, row in mp_subset.iterrows():
                cell_id = row["cell_id"]
                percentage = row["percentage"]

                if cell_id in cell_id_to_idx:
                    idx = cell_id_to_idx[cell_id]
                    weighted_pos += X_emb[idx] * percentage
                    total_weight += percentage

            if total_weight > 0:
                node_positions[node_name] = weighted_pos / total_weight
            else:
                node_positions[node_name] = X_emb.mean(axis=0)

        return node_positions

    def assign_cells_to_branches(self) -> pd.DataFrame:
        """
        根据 progressions 将细胞分配到分支，并计算 branch_lam

        每个细胞分配到其 percentage 最大的分支上
        节点名会根据 _milestone_to_node 映射进行转换

        Returns:
            DataFrame with columns: ['branch_id', 'branch_lam', 'from', 'to']
        """
        progressions = self.milestone_wrapper.progressions
        milestone_network = self.milestone_wrapper.milestone_network

        # 获取 milestone 到节点名的映射（如果存在）
        milestone_to_node = getattr(self, "_milestone_to_node", None)
        if milestone_to_node is None:
            # 如果还没构建映射，使用原始 milestone_id
            milestone_to_node = {mid: mid for mid in self.milestone_wrapper.id_list}

        # 为每个细胞找到最大 percentage 的分支
        result = []
        for cell_id in self.fadata.obs.index:
            cell_prog = progressions[progressions["cell_id"] == cell_id]

            if len(cell_prog) == 0:
                # 没有 progression 信息，跳过或使用默认值
                result.append(
                    {
                        "cell_id": cell_id,
                        "branch_id": None,
                        "branch_lam": 0.0,
                        "from": None,
                        "to": None,
                    }
                )
                continue

            # 找到 percentage 最大的行
            max_idx = cell_prog["percentage"].idxmax()
            max_row = cell_prog.loc[max_idx]

            from_milestone = max_row["from"]
            to_milestone = max_row["to"]
            percentage = max_row["percentage"]

            # 转换为节点名
            from_node = milestone_to_node.get(from_milestone, from_milestone)
            to_node = milestone_to_node.get(to_milestone, to_milestone)

            # 获取边长度
            edge_mask = ((milestone_network["from"] == from_milestone) & (milestone_network["to"] == to_milestone)) | (
                (milestone_network["from"] == to_milestone) & (milestone_network["to"] == from_milestone)
            )
            if edge_mask.any():
                edge_length = milestone_network.loc[edge_mask, "length"].iloc[0]
            else:
                edge_length = 1.0

            # branch_lam 是细胞在分支上的位置（0 到 edge_length）
            branch_lam = percentage * edge_length

            result.append(
                {
                    "cell_id": cell_id,
                    "branch_id": (from_node, to_node),
                    "branch_lam": branch_lam,
                    "from": from_node,
                    "to": to_node,
                }
            )

        df = pd.DataFrame(result).set_index("cell_id")
        return df

    def _project_cells_to_branches(self, node_positions: Dict, embedding_basis: str = "X_umap") -> Tuple[np.ndarray, np.ndarray]:
        """
        计算每个细胞到其分支的垂直距离 (branch_dist)

        Returns:
            branch_dist: 每个细胞到分支的距离
            branch_lam: 细胞在分支上的投影位置
        """
        X_emb = self.fadata.obsm[embedding_basis]
        n_cells = X_emb.shape[0]

        branch_dist = np.zeros(n_cells)
        branch_lam = np.zeros(n_cells)

        branch_info = self.assign_cells_to_branches()

        for idx, cell_id in enumerate(self.fadata.obs.index):
            cell_pos = X_emb[idx]

            if cell_id not in branch_info.index:
                continue

            cell_branch = branch_info.loc[cell_id]
            from_node = cell_branch["from"]
            to_node = cell_branch["to"]

            if from_node is None or to_node is None:
                continue

            # 获取分支两端点的位置
            pos_from = node_positions.get(from_node)
            pos_to = node_positions.get(to_node)

            if pos_from is None or pos_to is None:
                continue

            # 计算细胞到线段的投影
            proj_result = self._project_point_to_segment(cell_pos, pos_from, pos_to)
            branch_dist[idx] = proj_result["dist"]
            branch_lam[idx] = cell_branch["branch_lam"]

        return branch_dist, branch_lam

    @staticmethod
    def _project_point_to_segment(point: np.ndarray, seg_start: np.ndarray, seg_end: np.ndarray) -> Dict:
        """
        将点投影到线段上，返回投影点和距离

        Args:
            point: 目标点
            seg_start: 线段起点
            seg_end: 线段终点

        Returns:
            dict with 'proj_point', 'dist', 't' (参数化位置 0-1)
        """
        seg_vec = seg_end - seg_start
        seg_len_sq = np.dot(seg_vec, seg_vec)

        if seg_len_sq == 0:
            # 线段退化为点
            return {
                "proj_point": seg_start,
                "dist": np.linalg.norm(point - seg_start),
                "t": 0.0,
            }

        # 计算参数 t
        t = np.dot(point - seg_start, seg_vec) / seg_len_sq
        t = np.clip(t, 0, 1)

        # 投影点
        proj_point = seg_start + t * seg_vec
        dist = np.linalg.norm(point - proj_point)

        return {"proj_point": proj_point, "dist": dist, "t": t}

    def calculate_pseudotime(self, root_node: str = None) -> pd.Series:
        """
        计算从根节点出发的伪时间

        基于细胞所在分支和分支上的位置，计算到根节点的路径长度

        Args:
            root_node: 根节点，如果为 None 则自动选择第一个节点

        Returns:
            pd.Series: 每个细胞的伪时间
        """
        if root_node is None:
            root_node = self.milestone_wrapper.id_list[0]

        stream_tree = self.fadata.uns["stream_tree"]

        # 计算每个节点到根节点的路径长度
        node_to_root_dist = nx.shortest_path_length(stream_tree, source=root_node, weight="len")

        branch_info = self.assign_cells_to_branches()
        pseudotime = pd.Series(index=self.fadata.obs.index, dtype=float)

        for cell_id in self.fadata.obs.index:
            if cell_id not in branch_info.index:
                pseudotime[cell_id] = 0.0
                continue

            cell_branch = branch_info.loc[cell_id]
            from_node = cell_branch["from"]
            lam = cell_branch["branch_lam"]

            if from_node is None:
                pseudotime[cell_id] = 0.0
                continue

            # 计算 from_node 到根节点的距离
            dist_from_root = node_to_root_dist.get(from_node, 0.0)

            # 伪时间 = 到根节点的距离 + 在分支上的位置
            pseudotime[cell_id] = dist_from_root + lam

        return pseudotime

    def prepare_adata_for_stream(
        self,
        embedding_basis: str = "X_umap",
        root_node: str = None,
    ) -> FateAnnData:
        """
        为 stream.py 绘图准备 FateAnnData 对象

        这个方法会在原 FateAnnData 上添加必要的字段

        Args:
            embedding_basis: 用于计算节点位置的 embedding
            root_node: 根节点（milestone_id，用于确定绘图方向）

        Returns:
            准备好的 FateAnnData 对象
        """
        from .stream_extra import calculate_pseudotime as calc_pt

        logger.info(f"Preparing FateAnnData for stream plotting (model: {self.model_name})")

        if root_node is None:
            root_node = self.milestone_wrapper.id_list[0]

        # 1. 构建 stream_tree（包含 'root' 别名节点）
        stream_tree = self.build_stream_tree(root_milestone=root_node)
        self.fadata.uns["stream_tree"] = stream_tree
        logger.debug(f"Stream tree built with {len(stream_tree.nodes())} nodes, {len(stream_tree.edges())} edges")

        # 2. 计算节点在 embedding 中的位置
        node_positions = self._compute_node_positions_from_progressions(embedding_basis)

        # 3. 将节点位置保存到 stream_tree 的节点属性中
        # 这是 stream_extra 函数需要的
        pos_attr = f"X_{embedding_basis.replace('X_', '')}_pos"
        for node_id, pos in node_positions.items():
            if node_id in stream_tree.nodes():
                stream_tree.nodes[node_id][pos_attr] = pos

        # 4. 分配细胞到分支并计算投影
        branch_df = self.assign_cells_to_branches()
        self.fadata.obs["branch_id"] = branch_df["branch_id"]
        self.fadata.obs["branch_id_alias"] = branch_df["branch_id"].apply(
            lambda x: (
                (stream_tree.nodes[x[0]]["label"], stream_tree.nodes[x[1]]["label"])
                if x is not None and x[0] in stream_tree.nodes() and x[1] in stream_tree.nodes()
                else None
            )
        )

        # 5. 计算 branch_dist 和 branch_lam
        branch_dist, branch_lam = self._project_cells_to_branches(node_positions, embedding_basis)
        self.fadata.obs["branch_dist"] = branch_dist
        self.fadata.obs["branch_lam"] = branch_lam

        # 6. 计算 pseudotime（使用 stream_extra 的函数）
        calc_pt(self.fadata)

        # 7. 设置工作目录
        if "workdir" not in self.fadata.uns:
            self.fadata.uns["workdir"] = "."

        logger.info("FateAnnData prepared successfully for stream plotting")
        return self.fadata

    def plot_stream_sc(
        self,
        root: str = "root",
        color: List[str] = None,
        fig_size: Tuple = (7, 4.5),
        fig_legend_ncol: int = 1,
        save_fig: bool = False,
        fig_path: str = None,
        fig_format: str = "pdf",
        **kwargs,
    ):
        """
        绘制 stream 单细胞级别的图（subway map）

        Args:
            root: 根节点标识
            color: 要上色的列表（obs 中的列名或 var 中的基因名）
            fig_size: 图表大小
            fig_legend_ncol: 图例的列数
            save_fig: 是否保存图表
            fig_path: 保存路径
            fig_format: 保存格式
            **kwargs: 传递给 stream.plot_stream_sc 的其他参数
        """
        from .stream import plot_stream_sc as _plot_stream_sc

        # 使用默认的上色列
        if color is None:
            cluster_key = self.fadata.prior_information.get("cluster")
            if cluster_key and cluster_key in self.fadata.obs.columns:
                color = [cluster_key]
            else:
                color = ["group"] if "group" in self.fadata.obs.columns else [self.fadata.obs.columns[0]]

        # 自动将 categorical 类型转换为字符串，同时保存原始类别顺序
        original_categories = {}
        for col in color:
            if col in self.fadata.obs.columns:
                if pd.api.types.is_categorical_dtype(self.fadata.obs[col]):
                    # 保存原始类别顺序（用于颜色映射）
                    original_categories[col] = list(self.fadata.obs[col].cat.categories)
                    logger.debug(f"Converting categorical column '{col}' to string")
                    self.fadata.obs[col] = self.fadata.obs[col].astype(str)

        # 转换 scanpy 颜色格式 (_colors 数组) 为 stream.py 格式 (_color 字典)
        for col in color:
            if col in self.fadata.obs.columns:
                colors_key = f"{col}_colors"  # scanpy 格式
                color_key = f"{col}_color"  # stream.py 格式
                if colors_key in self.fadata.uns and color_key not in self.fadata.uns:
                    colors_array = self.fadata.uns[colors_key]
                    # 使用保存的原始类别顺序，或从当前唯一值获取
                    if col in original_categories:
                        categories = original_categories[col]
                    else:
                        categories = list(self.fadata.obs[col].unique())
                    # 创建字典映射
                    if len(colors_array) >= len(categories):
                        color_dict = {cat: colors_array[i] for i, cat in enumerate(categories)}
                        self.fadata.uns[color_key] = color_dict
                        logger.debug(f"Converted {colors_key} to {color_key} dict format: {color_dict}")

        if fig_path is None:
            fig_path = self.fadata.uns.get("workdir", ".")

        logger.info(f"Plotting stream_sc with color: {color}")

        # 调用本地的绘图函数
        figs = _plot_stream_sc(
            self.fadata,
            root=root,
            color=color,
            fig_size=fig_size,
            fig_legend_ncol=fig_legend_ncol,
            save_fig=save_fig,
            fig_path=fig_path,
            fig_format=fig_format,
            return_fig=True,
            **kwargs,
        )

        return figs

    def plot_stream(
        self,
        root: str = "root",
        color: List[str] = None,
        fig_size: Tuple = (7, 4.5),
        save_fig: bool = False,
        fig_path: str = None,
        fig_format: str = "pdf",
        **kwargs,
    ):
        """
        绘制 stream 密度级别的图

        Args:
            root: 根节点标识
            color: 要上色的列表
            fig_size: 图表大小
            save_fig: 是否保存图表
            fig_path: 保存路径
            fig_format: 保存格式
            **kwargs: 传递给 stream.plot_stream 的其他参数
        """
        from .stream import plot_stream as _plot_stream

        if color is None:
            cluster_key = self.fadata.prior_information.get("cluster")
            if cluster_key and cluster_key in self.fadata.obs.columns:
                color = [cluster_key]
            else:
                color = ["group"] if "group" in self.fadata.obs.columns else [self.fadata.obs.columns[0]]

        if fig_path is None:
            fig_path = self.fadata.uns.get("workdir", ".")

        logger.info(f"Plotting stream with color: {color}")

        figs = _plot_stream(
            self.fadata,
            root=root,
            color=color,
            fig_size=fig_size,
            save_fig=save_fig,
            fig_path=fig_path,
            fig_format=fig_format,
            return_fig=True,
            **kwargs,
        )

        return figs


def plot_stream_from_fateanndata(
    fadata: FateAnnData,
    model_name: str = None,
    plot_type: str = "sc",  # 'sc' 或 'stream'
    embedding_basis: str = "X_umap",
    root: str = "root",
    color: List[str] = None,
    save_fig: bool = False,
    fig_path: str = None,
    **kwargs,
) -> List[plt.Figure]:
    """
    便捷函数：直接从 FateAnnData 绘制 stream 图

    Args:
        fadata: FateAnnData 对象（需要已调用 add_trajectory）
        model_name: 轨迹模型名称
        plot_type: 绘图类型 ('sc' 或 'stream')
        embedding_basis: 用于节点位置的 embedding 键
        root: 根节点标识（用于树中设置 'root' 节点）
        color: 上色列表
        save_fig: 是否保存
        fig_path: 保存路径
        **kwargs: 其他参数

    Returns:
        图表列表
    """
    # 创建适配器
    adapter = StreamPlotAdapter(fadata, model_name=model_name)

    # 准备数据
    adapter.prepare_adata_for_stream(embedding_basis=embedding_basis)

    # 绘图
    if plot_type == "sc":
        figs = adapter.plot_stream_sc(root=root, color=color, save_fig=save_fig, fig_path=fig_path, **kwargs)
    elif plot_type == "stream":
        figs = adapter.plot_stream(root=root, color=color, save_fig=save_fig, fig_path=fig_path, **kwargs)
    else:
        raise ValueError(f"Unknown plot_type: {plot_type}")

    return figs
