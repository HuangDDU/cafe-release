import networkx as nx

from .add_color import add_milestone_color


def plot_topology(fadata, ax=None, nx_draw_kwrags={}, layout="dot"):
    milestone_wrapper = fadata.milestone_wrapper
    milestone_id_list = milestone_wrapper.id_list
    milestone_network = milestone_wrapper.milestone_network.copy()
    is_directed = milestone_network["directed"].any()

    # color and position fo milestone
    milestone_color_list = add_milestone_color(len(milestone_id_list))  # color rgb
    milestone_color_dict = dict(zip(milestone_id_list, milestone_color_list))
    milestone_network["len"] = milestone_network["length"]  # 边权值对于权值的影响
    G = nx.from_pandas_edgelist(
        milestone_network,
        source="from",
        target="to",
        edge_attr=["len"],
        create_using=nx.DiGraph if is_directed else nx.Graph,
    )
    if layout in ["dot", "neato"]:
        pos = nx.nx_agraph.graphviz_layout(G, prog=layout)  # dot for layer structure, neato for weighted graph structure
    else:
        pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color=[milestone_color_dict[node] for node in G.nodes], ax=ax, **nx_draw_kwrags)
