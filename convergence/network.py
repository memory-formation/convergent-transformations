
from typing import Literal
import networkx as nx
import pandas as pd
import numpy as np
from scipy.sparse.csgraph import laplacian, reverse_cuthill_mckee
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plot_connection(
    start,
    end,
    ax,
    color="black",
    alpha=1,
    linewidth=1,
    zorder=0,
    direction: Literal["->", "<-", "-"] = "-",
    arrow_size=10
):
    if direction == "-":
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            zorder=zorder,
        )
    else:
        # Reverse if direction is "<-"
        if direction == "<-":
            start, end = end, start
            direction = "->"

        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops=dict(
                arrowstyle=direction,
                color=color,
                alpha=alpha,
                linewidth=linewidth,
                mutation_scale=arrow_size,
                zorder=1000,
            ),
            zorder=zorder,
        )
 
def efficient_iterative_max_spanning_trees(original_graph_nx, num_trees, similarity_attr='similarity'):
    # Create a working graph with 'cost' for MST and keep original similarity
    working_graph = nx.Graph()
    working_graph.add_nodes_from(original_graph_nx.nodes())
    for u, v, data in original_graph_nx.edges(data=True):
        similarity = data.get(similarity_attr, 0.0) # Get similarity, default to 0
        cost = 1 - similarity #abs(similarity)
        working_graph.add_edge(u, v, cost=cost, original_similarity=similarity)

    final_backbone_graph = nx.Graph()
    final_backbone_graph.add_nodes_from(original_graph_nx.nodes())

    for i in range(num_trees):
        if working_graph.number_of_edges() == 0:
            # print(f"No more edges in working graph at iteration {i+1}.")
            break

        current_mst_edges = []
        # Handle potentially disconnected graph by finding MST for each component
        if not nx.is_connected(working_graph): # This check can be slow on very large graphs
            # print(f"Graph is disconnected in iteration {i+1}. Processing components.")
            for component_nodes in list(nx.connected_components(working_graph)): # list() for safe iteration if modifying
                subgraph = working_graph.subgraph(component_nodes)
                if subgraph.number_of_edges() > 0:
                    component_mst = nx.minimum_spanning_tree(subgraph, weight='cost', algorithm='kruskal')
                    current_mst_edges.extend(component_mst.edges(data=True))
        elif working_graph.number_of_edges() > 0:
            mst = nx.minimum_spanning_tree(working_graph, weight='cost', algorithm='kruskal')
            current_mst_edges = list(mst.edges(data=True))

        if not current_mst_edges:
            # print(f"No MST found in iteration {i+1}. Stopping.")
            break

        added_new_edge_in_iteration = False
        edges_to_penalize = []
        for u, v, mst_edge_data in current_mst_edges:
            if not final_backbone_graph.has_edge(u, v):
                # Retrieve original similarity stored in the working_graph edge
                original_sim = working_graph[u][v]['original_similarity']
                final_backbone_graph.add_edge(u, v, **{similarity_attr: original_sim, 'iteration_added': i+1})
                added_new_edge_in_iteration = True

            # Store edges whose weights need to be updated in the working_graph
            edges_to_penalize.append((u,v))

        # Penalize edges in the working_graph
        for u,v in edges_to_penalize:
            if working_graph.has_edge(u,v): # Ensure edge still exists
                working_graph[u][v]['cost'] = float('inf')
                

        if not added_new_edge_in_iteration and i > 0: # No new edges added, and not the first tree
            # print(f"No new unique edges added in iteration {i+1}. Stopping.")
            break

    return final_backbone_graph





def reorder_matrix_spectral_simplified(
    df: pd.DataFrame,
    use_normalized_laplacian: bool = True,
    interpret_negative_as_weak_connection: bool = True,
) -> pd.DataFrame:
    node_names = df.index
    conn_matrix_original = df.to_numpy()  # Keep original values for reconstruction

    if interpret_negative_as_weak_connection:
        # For Laplacian, typically non-negative weights are assumed.
        # This clips negative values to 0.
        # If your negative values have specific meaning for clustering (e.g., strong repulsion),
        # consider df.abs() or other preprocessing before calling this function,
        # and set interpret_negative_as_weak_connection=False.
        processed_conn_matrix_for_laplacian = np.maximum(conn_matrix_original, 0)
    else:
        processed_conn_matrix_for_laplacian = conn_matrix_original

    L = laplacian(processed_conn_matrix_for_laplacian, normed=use_normalized_laplacian)
    eigenvalues, eigenvectors = eigh(L)  # eigh for dense symmetric matrices

    # Fiedler vector is the eigenvector for the 2nd smallest eigenvalue
    fiedler_vector = eigenvectors[:, 1]
    new_order_indices = np.argsort(fiedler_vector)

    # Reconstruct the DataFrame with the new order using original matrix values
    reordered_df = pd.DataFrame(
        conn_matrix_original[np.ix_(new_order_indices, new_order_indices)],
        index=node_names[new_order_indices],
        columns=node_names[new_order_indices],
    )
    return reordered_df


def reorder_matrix_rcm_simplified(df: pd.DataFrame) -> pd.DataFrame:
    node_names = df.index
    conn_matrix_np = df.to_numpy()

    # RCM operates on the sparsity structure of the graph
    graph_sparse = csr_matrix(conn_matrix_np)
    rcm_order_indices = reverse_cuthill_mckee(graph_sparse, symmetric_mode=True)

    # Reconstruct the DataFrame with the new order
    reordered_df = pd.DataFrame(
        conn_matrix_np[np.ix_(rcm_order_indices, rcm_order_indices)],
        index=node_names[rcm_order_indices],
        columns=node_names[rcm_order_indices],
    )
    return reordered_df


def plot_graph(
        G: nx.Graph,
        df_nodes: pd.DataFrame,
        df_depths_edges: pd.DataFrame = None,
        cmap_node: str = "PiYG_r",
        cmap_edge: str = "RdBu_r",
        vrange_edge = (-0.2, 0.2),
        vrange_node = (0.2, 0.8),
        figsize: tuple = (6, 8),
        ax: plt.Axes = None,
        figure_title: str = None,
        layout_adjustments: dict = None,
        seed: int = 2,
        weight_column: str="weight",
        similarity_column: str = "similarity",
        gravity: float = 10,
        node_size_base: float = 3.0,
        node_size_multiplier: float = 250.0,
        node_label_fontsize: int = 10,
        node_label_dy_offset: float = 0.02,
        node_edge_linewidth: float = 1.5,
        node_value_column: str = "vision_depth",
        edge_lw_base: float = 1.0,
        edge_lw_multiplier: float = 2.0,
        arrow_size: int = 18,
        iteration_multiplier: int = 1,  # Used to adjust edge width based on iteration
        custom_layout = None,
):
    # Get figure
    if ax is None:
        fig, current_ax = plt.subplots(figsize=figsize)
    else:
        current_ax = ax
        fig = current_ax.figure
    current_ax.axis("off")
    if figure_title:
        current_ax.set_title(figure_title)
    
    # Get cmaps of nodes and edges
    cmap_node = plt.get_cmap(cmap_node)
    norm_node = mcolors.Normalize(vmin=vrange_node[0], vmax=vrange_node[1])
    cmap_edge = plt.get_cmap(cmap_edge)
    norm_edge = mcolors.Normalize(vmin=vrange_edge[0], vmax=vrange_edge[1])
    
    # Obtain the layout
    if custom_layout is not None:
        layout = custom_layout(G)
    else:
        layout = nx.forceatlas2_layout(G, seed=seed, weight=weight_column, dim=2, gravity=gravity)
    if layout_adjustments is not None:
        for node, adjust in layout_adjustments.items():
            layout[node] += adjust

    df_nodes = df_nodes.copy().set_index("name")

    for node_name, (x, y) in layout.items():
        node_data = df_nodes.loc[node_name] 
        score = node_data[similarity_column]
        norm_score = norm_edge(score)
        depth = node_data[node_value_column]
        node_s = (node_size_base + node_size_multiplier * norm_score**2)
        node_v = norm_node(depth)
        node_display_color = cmap_node(node_v)
        node_border_color = node_data.area_color
        
        current_ax.scatter(x, y,
            s=node_s,
            color=node_display_color,
            alpha=1,
            zorder=int(500 + 300 * norm_score),
            edgecolors=node_border_color,
            linewidths=node_edge_linewidth,
            marker="o",
        )
        node_label_fontsize_node = node_label_fontsize
        coef = 0.7
        node_label_fontsize_node = (
            node_label_fontsize if norm_score > 0.6 else coef * node_label_fontsize
        )
        node_label_dy_offset_node = (
            node_label_dy_offset if norm_score > 0.6 else coef * node_label_dy_offset
        )
        current_ax.text(
            x,
            y + node_label_dy_offset_node,
            node_name,
            fontsize=node_label_fontsize_node,
            ha="center",
            zorder=1000,
        )

    # Plot edges
    for node1, node2, edge_data in G.edges(data=True):
        start_pos = layout[node1]
        end_pos = layout[node2]
        
        edge_score = edge_data.get(similarity_column, 0.0)
        iteration_added = edge_data.get("iteration_added", 1)
        norm_edge_score = norm_edge(edge_score)
        lw = (edge_lw_base + edge_lw_multiplier * norm_edge_score) / (1+iteration_multiplier*(iteration_added-1))
        lw = (edge_lw_base + edge_lw_multiplier * 0.8) / (
            1 + iteration_multiplier * (iteration_added - 1)
        )  # Patch
        edge_color = cmap_edge(norm_edge_score)
        direction = "-"

        if df_depths_edges is not None:
            row = df_depths_edges.query("name_x == @node1 and name_y == @node2")
            if not row.empty:
                if row.p_corrected.values[0] <= 0.05:
                    t_stat = row.t_stat.values[0]
                    if t_stat < 0:
                        direction = "->"
                    elif t_stat > 0:
                        direction = "<-"
                

        plot_connection(
            start_pos,
            end_pos,
            current_ax,
            color=edge_color,
            alpha=1,
            linewidth=lw,
            zorder=int(norm_edge_score * 100),
            direction=direction,
            arrow_size=arrow_size,
        )
        
    return fig, current_ax

