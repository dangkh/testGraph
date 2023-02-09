import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx

seed = 13648  # Seed random number generators for reproducibility
G = nx.random_k_out_graph(20, 3, 0.5, seed=seed)
G = G.to_undirected()
# G = nx.star_graph(20)
pos = nx.spring_layout(G, seed=63)  # Seed layout for reproducibility
colors = range(G.number_of_edges())
options = {
    "node_color": "#A0CBE2",
    "edge_color": colors,
    "width": 4,
    "edge_cmap": plt.cm.Blues,
    "with_labels": False,
}
nodes = nx.draw_networkx_nodes(G,pos,node_color='#A0CBE2')
edges = nx.draw_networkx_edges(G,pos,edge_color=colors,width=4,
                               edge_cmap=plt.cm.Blues)
print(edges)
plt.colorbar(edges)
ax = plt.gca()
ax.set_axis_off()
plt.show()
