import os
import json
import numpy as np
from collections import deque
import networkx as nx
import matplotlib.pyplot as plt

import operator_calcs as oc


##### Woriing with JSON #####
class DependancyNode():
    def __init__(self, oppid, node, parents, input_shapes, output_shapes):
        self.oppid = oppid
        self.func = 'null' if node['op'] == "null" else node['name']
        self.parents = parents
        self.input_shapes = input_shapes
        self.output_shapes = output_shapes
        self.opp = self._find_opp(node['attrs']['func_name']) if 'attrs' in node else 'null'
        self.time = self._unfsed_graph_time(node) # (CPU, PHU) time
        self.hardware = None

    def __hash__(self):
        return hash(self.oppid)

    def __eq__(self, other):
        if isinstance(other, DependancyNode):
            return self.oppid == other.oppid
        return False

    def __str__(self):
        return f"""
        {self.oppid=}
        {self.func=}
        {self.opp=}
        {self.input_shapes=}
        {self.output_shapes=}
        {self.hardware=}
        {self.time=}"""

    def _find_opp(self, func_name):
        name_parts = func_name.split('_')
        for part in reversed(name_parts):
            try:
                int(part)
                continue
            except:
                return part

    def _unfsed_graph_time(self, node):
        # takes node, returns time for each hardware choice
        if 'attrs' in node:
            return [oc.opp_time_func(self.opp, self.input_shapes, self.output_shapes, run_hardware) for run_hardware in hardware_config]
        return (0, 0)

class DependancyGraph():
    def __init__ (self, raw_json):
        self.raw_json = raw_json
        self.node_list = self.create_nodes()
        self.optimization = {'always_CPU': self._always_CPU, 'min': self._min}

    def create_nodes(self):
        ajusted_shapes = []
        split_shift = 0
        for index, node in enumerate(self.raw_json["nodes"]):
            ajusted_shapes.append(self.raw_json['attrs']['shape'][1][index + split_shift])
            if 'split' in node['name']:
                split_shift += 2

        nodes = []
        for index, node in enumerate(self.raw_json["nodes"]):
            oppid = index
            num_output = int(node['attrs']['num_outputs']) if 'attrs' in node else 1
            parents = [shape_idx[0] for shape_idx in node['inputs']]
            input_shapes = [ajusted_shapes[shape_idx[0]] for shape_idx in node['inputs']]
            output_shapes = [ajusted_shapes[index] for i in range(num_output)]
            nodes.append(DependancyNode(oppid, node, parents, input_shapes, output_shapes))
        return nodes

    # Create the Adjancy Matrix for dependancy
    def creat_adj_matrix_json(self):
        dependancys = []
        for node_index, node in enumerate(self.raw_json['nodes']):
            inputs = node.get('inputs', [])
            for inp in inputs: # where each input is an index to another node.
                dependancys.append((inp[0], node_index)) # (1, 2) where 2 takes 1's output

        num_nodes = (len(self.raw_json['nodes']))
        adj_matrix = np.zeros((num_nodes, num_nodes))
        for dep in dependancys:
            adj_matrix[dep[0]][dep[1]] = 1
        return adj_matrix

    def creat_adj_matrix_node_list(self):
        dependancys = []
        for node in self.node_list:
            inputs = node.parents
            for inp in inputs: # where each input is an index to another node.
                dependancys.append((inp, node.oppid, self.bit_transfer(node))) # (1, 2) where 2 takes 1's output
                G.addEdge(inp, node.oppid, self.bit_transfer(node))
                # if (550 <= node.oppid <= 638):

        num_nodes = (len(self.node_list))
        adj_matrix = np.zeros((num_nodes, num_nodes))
        for dep in dependancys:
            adj_matrix[dep[0]][dep[1]] = dep[2]
        return adj_matrix

    def matrix_to_CSR(self, matrix):
        data = []
        column_offset = []
        node_row_ptr = [0]
        row_prt_total = 0
        for row_num, row in enumerate(matrix):
            for col_num, cell in enumerate(row):
                if cell:
                    data.append(cell)
                    column_offset.append(col_num)
                    row_prt_total += 1
            node_row_ptr.append(row_prt_total)
        return data, column_offset, node_row_ptr

    # node opps
    def write_graph(self):
        with open('custon_graph.txt', "w") as f:
            for node in self.node_list:
                if node.opp != 'null':
                    f.write(f'{node.func}\n')

    def create_cost_vec(self):
        CPU_cost = [node.time[0] if isinstance(node.time, list) else node.time for node in self.node_list]
        PHU_cost = [node.time[1] if isinstance(node.time, list) else node.time for node in self.node_list]
        return CPU_cost, PHU_cost

    def get_opp_spread(self):
        cycles_per_op = {}
        for node in self.node_list:
            cycles_per_op.setdefault(node.opp, 0)
            cycles_per_op[node.opp] += node.time[0] # CPU
        return cycles_per_op

    # Time
    def total_time(self, optimization):
        # Traverse the graph making time considerations
        return 0

    def bit_transfer(self, node):
        total = 0
        for shape in node.output_shapes:
            total += oc.ten_elm(shape)
        return total

    def _always_CPU(self, node, CPU_time, PHU_time):
        node.hardware = 'CPU'
        return CPU_time

    def _min(self, node, CPU_time, PHU_time):
        if CPU_time < np.inf and PHU_time < np.inf:

            if CPU_time < PHU_time:
                node.hardwer = 'CPU'
                return CPU_time
            else:
                node.hardwer = 'PHU'
                return PHU_time
        else:
            node.hardware = 'CPU'
            return CPU_time

    # scheduling
    def kahn_topo_sort(self, graph):
        '''
        produes a liner order obeying DAG
        graph(np adjancy matrix): graph to sort
        order(list): liner order
        layers_list(list of lists): each entry is a dependancy layer
        '''
        node_indegree = {}
        for idx in range(len(graph)):
            node_indegree[idx] = np.sum(graph[:, idx] != 0)


        que = []
        order = []
        for node, val in node_indegree.items():
            if val == 0:
                que.append(node)

        layer = 0
        layers_dic = {}
        while que:
            layer += 1
            layers_dic[layer] = []
            for _ in range(len(que)):
                cur_node = que.pop(0)
                order.append(cur_node)
                layers_dic[layer].append(cur_node)
                for next_node in np.where(graph[cur_node])[0]:
                    node_indegree[next_node] -= 1
                    if node_indegree[next_node] == 0:
                        que.append(next_node)

        assert any(node_indegree.values()) == False

        layers_list =  [val for (key, val) in layers_dic.items()]
        return order, layers_list

    def kahn_topo_sort_working(self, graph):
        '''
        produes a liner order obeying DAG
        graph(np adjancy matrix): graph to sort

        order: liner working order for each node
        working_layers_list: nodes that can work on each layer
        layer_count: the amount of layers each node can work on
        '''
        node_indegree = {}
        node_parents = {}
        node_outdegree = {'START': np.inf}
        for idx in range(len(graph)):
            node_indegree[idx] = np.sum(graph[:, idx] != 0)
            node_outdegree[idx] = np.sum(graph[idx, :] != 0)
            node_parents[idx] = []


        que = []
        order = []
        layer_count = {}
        for node, val in node_indegree.items():
            if val == 0:
                que.append(( ['START'] ,node))
                layer_count[node] = 0

        layer = 0
        layers_dic = {}
        while que:
            layer += 1
            layers_dic[layer] = set()

            for _ in range(len(que)):
                par_nodes, cur_node = que.pop(0)
                for par in par_nodes:
                    node_outdegree[par] -= 1

                order.append(cur_node)
                layers_dic[layer].add(cur_node)
                for next_node in np.where(graph[cur_node])[0]:
                    node_indegree[next_node] -= 1
                    node_parents[next_node].append(cur_node)
                    if node_indegree[next_node] == 0:
                        que.append((node_parents[next_node], next_node))
                        layer_count[next_node] = 0

            for working in order:
                if node_outdegree[working] != 0:
                    layers_dic[layer].add(working)
                    layer_count[working] += 1

        assert any(node_indegree.values()) == False

        layers_list =  [val for (key, val) in layers_dic.items()]
        return order, layers_list, layer_count



read_json_path = '/home/rjtomich/photonic_compiler/model_to_graph/gpt2_graph.json'
# read_json_path = '/home/rjtomich/photonic_compiler/model_to_graph/bert-base-uncased_graph.json'
# read_json_path = '/home/rjtomich/photonic_compiler/Pytorch-LeNet/simple_LeNet_graph.json'
with open(read_json_path)  as json_file:
    raw_json = json.load(json_file) # returns json file as dict

# Constants
hardware_config = ["run_cpu", "run_phu"]
optimization =('always_CPU', 'min')

E_PH_BIT_COST = 3 /oc.PHU_CLOCK_SPEED
BITS_PER_FLOAT = 32
E_PH_COST = E_PH_BIT_COST * BITS_PER_FLOAT



# making graph
# graph = DependancyGraph(raw_json)
# CPU_cost, PHU_cost = graph.create_cost_vec()
# adj_matrix = graph.creat_adj_matrix()

# print(graph.node_list[598])

# order, layers_list = graph.kahn_topo_sort(adj_matrix)
# working_order, working_layers_list, working_layer_count = graph.kahn_topo_sort_working(adj_matrix)

# for node in graph.node_list:
#     if node.opp == "dense":
#         print (node.input_shapes)
#         print(node.output_shapes)
#         print(node.time)



## Timing
# print(f"always_CPU: {graph.total_time('always_CPU')}")
# print(f"min: {graph.total_time('min')}")

## Visualization

class GraphVisualization:

    def __init__(self):
        self.visual = []

    def addEdge(self, a, b, weight=1.0):
        temp = [a, b, weight]
        self.visual.append(temp)

    def visualize(self, layout='spring', filename='graph.png'):
        G = nx.DiGraph()
        G.add_edges_from(self.visual)

        # Choose a layout algorithm
        if layout == 'spring':
            pos = nx.spring_layout(G, k=0.1, iterations=10)
        elif layout == 'shell':
            pos = nx.shell_layout(G)
        elif layout == 'spectral':
            pos = nx.spectral_layout(G, scale=1.0)
        elif layout == 'kk':
            pos = nx.kamada_kawai_layout(G)

        else:
            raise ValueError("Unknown layout type. Choose from 'spring', 'shell', or 'spectral'.")

        # Create a larger figure
        plt.figure(figsize=(30, 30))

        # Draw the graph with adjusted settings
        nx.draw(G, pos, node_size=100, width=0.5, with_labels=False, node_color='blue', edge_color='black')

        # Add labels to the nodes
        labels = {node: str(node) for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=5)

        # Save the plot to a file
        plt.savefig(filename, dpi=300)  # Adjust dpi as needed
        plt.close()


G = GraphVisualization()

graph = DependancyGraph(raw_json)
adj = graph.creat_adj_matrix_node_list()

G.visualize(layout='kk', filename='network.png') ####

# G.visualize(layout='shell', filename='network.png')
# G.visualize(layout='spring', filename='network.png')
# G.visualize(layout='spectral', filename='network.png')


# # Plotting the adjacency matrix
# plt.figure(figsize=(6, 6))
# plt.imshow(adj, cmap='binary', interpolation='none')
# plt.title('GPT2 Adjacency Matrix')
# plt.colorbar()
# plt.savefig('network.png', dpi=300)
# plt.close()
