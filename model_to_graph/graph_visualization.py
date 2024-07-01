import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

class GraphVisualization:

    def __init__(self):
        self.visual = []

    def addEdge(self, a, b, weight = 1):
        temp = [a, b, weight]
        self.visual.append(temp)

    def normalize(self, value, min_value, max_value):
        assert min_value != max_value
        return max( ((value - min_value) / (max_value - min_value)), 0.25)

    def visualize(self, layout='spring', filename='graph.png'):
        G = nx.DiGraph()
        # G.add_edges_from(self.visual)

        weight_range = 0
        for edge in self.visual:
            a, b, weight = edge
            weight_range = max (weight_range, weight)
            G.add_edge(a, b, weight=weight)

        # Choose a layout algorithm
        if layout == 'spring':
            # pos = nx.spring_layout(G, k=0.1, iterations=10)
            pos = nx.spring_layout(G, weight='weight', k=0.1, iterations=10)
        elif layout == 'shell':
            pos = nx.shell_layout(G)
        elif layout == 'spectral':
            pos = nx.spectral_layout(G, scale=1.0)
        elif layout == 'kk':
            lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
            pos = nx.kamada_kawai_layout(G, dist=lengths)
            # pos = nx.kamada_kawai_layout(G)

        else:
            raise ValueError("Unknown layout type. Choose from 'spring', 'shell', or 'spectral'.")

        plt.figure(figsize=(15, 15))

        edges = G.edges(data=True)
        weights = [self.normalize(edge[2]['weight'],0,weight_range) for edge in edges]

        nx.draw(G, pos, node_size=100, width=weights, with_labels=False, node_color='lightblue', edge_color='black')

        # Add labels to the nodes
        labels = {node: str(node) for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=10)

        # Save the plot to a file
        plt.savefig(filename, dpi=300)  # Adjust dpi as needed
        plt.close()


# G = GraphVisualization()

# graph = sg.DependancyGraph(sg.raw_json)
# adj = graph.creat_adj_matrix_node_list()

# G.visualize(layout='kk', filename='network.png') #### cant use weights
# G.visualize(layout='spring', filename='network.png')
# G.visualize(layout='shell', filename='network.png')
# G.visualize(layout='spectral', filename='network.png')


# # Plotting the adjacency matrix
# plt.figure(figsize=(6, 6))
# plt.imshow(adj, cmap='binary', interpolation='none')
# plt.title('GPT2 Adjacency Matrix')
# plt.colorbar()
# plt.savefig('network.png', dpi=300)
# plt.close()


def adj_to_graph(graph, ax, save=False, layout = 'shell', title ='Graph Visualization from Adjacency Matrix'):
    vectorized_function = np.vectorize(lambda x: 1 if x is not None else 0)

    adj_matrix = vectorized_function(graph.adj_matrix)

    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)

    labels = {}
    colors = {}
    for idx, node in enumerate(G.nodes):
        labels[node] = graph.stack_list[idx].oppid
        if graph.stack_list[idx].func_selection == 1:
            colors[node] = 'lightcoral'
        else:
            colors[node] = 'lightblue'


    nx.set_node_attributes(G, labels, 'label')


    if layout == 'shell':
        pos = nx.shell_layout(G)
    elif layout == 'spectral':
        pos = nx.spectral_layout(G, scale=1.0)
    elif layout == 'kk':
        lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
        pos = nx.kamada_kawai_layout(G, dist=lengths)
    elif layout == 'spring':
        # pos = nx.spring_layout(G, k=0.1, iterations=10)
        pos = nx.spring_layout(G, weight='weight', k=0.1, iterations=10)

    nx.draw(G, pos, with_labels=True,labels=labels, node_color=[colors[node] for node in G.nodes], edge_color='gray', node_size=200, font_size=7, ax=ax)
    ax.set_title(title)
    ax.set_aspect('equal')
