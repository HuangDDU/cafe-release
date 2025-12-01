import pandas as pd
import plotly.express as px


def plot_embedding_plotly(adata, color="clusters", basis="umap", obs_attribute_list=[]):
    # 使用plotly交互查看，方便选择细胞

    # 提取降维坐标
    embbedding_key = f"X_{basis}"
    x_name = f"{basis}1"
    y_name = f"{basis}2"
    emb_df = pd.DataFrame(data=adata.obsm[embbedding_key][:, :2], columns=[x_name, y_name], index=adata.obs_names)

    # 在obs中删除列已有的x,y列名
    obs = adata.obs.copy()
    for col in [x_name, y_name]:
        if col in obs.columns:
            obs = obs.drop(columns=[col])

    emb_df = emb_df.join(obs)

    # 绘制交互式 UMAP 图
    fig = px.scatter(
        emb_df,
        x=x_name,
        y=y_name,
        color=color,  # 按聚类着色（替换为你的 obs 列名）
        hover_name=adata.obs_names,  # 主悬停标题为细胞名称
        hover_data={col: True for col in obs_attribute_list},  # 显示其他信息
        width=800,
        height=600,
    )
    # 显示图表
    fig.show()
