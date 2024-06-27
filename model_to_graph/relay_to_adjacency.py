import os
import json
import numpy as np
from collections import deque
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


import operator_calcs as oc
import graph_visualization as gv


##### Woriing with JSON #####
class DependancyNode():
    def __init__(self, oppid, node, parents, input_shapes, output_shapes):
        self.oppid = oppid
        self.func = 'null' if node['op'] == "null" else node['name']
        self.parents = parents
        self.input_shapes = input_shapes
        self.output_shapes = output_shapes
        self.opp = self._find_opp(node['attrs']['func_name']) if 'attrs' in node else 'null'
        self.time = self._unfsed_node_time(node) #(CPU, PHU)
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
        '''
        func_name(srt) - whole tvm function name
        returns (srt) - last collection of letters, which is the name
        '''
        name_parts = func_name.split('_')
        for part in reversed(name_parts):
            try:
                int(part)
                continue
            except:
                return part

    def _unfsed_node_time(self, node):
        '''
        node(DependancyNode object)
        returns(tuple) - the time tuple for each node for each hardware configuration
        '''
        if 'attrs' in node:
            return [oc.opp_time_func(self.opp, self.input_shapes, self.output_shapes, hardware_config) for hardware_config in hardware_config_list]
        return [0, np.inf]

class DependancyGraph():
    def __init__ (self, raw_json):
        self.raw_json = raw_json
        self.node_list = self.create_nodes()
        self.hardware_selection_funcs = {'always_CPU': self._always_CPU, 'naive_min': self._naive_min, 'always_PHU': self._always_PHU}

    def create_nodes(self):
        '''
        Uses raw_json of the DependancyGraph to make list of DependancyNode objects
        '''
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
        '''
        Creates an adjancy matrix of the dependencies using raw_json
        '''
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
        '''
        Creates an adjancy matrix of the dependencies using node_list
        '''
        dependancys = []
        for node in self.node_list:
            inputs = node.parents
            for inp in inputs: # where each input is an index to another node.
                dependancys.append((inp, node.oppid, self.bit_transfer(self.node_list[inp]))) # (1, 2) where 1's output are 2's inputs
                # G.addEdge(inp, node.oppid, self.bit_transfer(node))
                if 950< node.oppid < 1108:
                    G.addEdge(inp, node.oppid)

        num_nodes = (len(self.node_list))
        adj_matrix = np.zeros((num_nodes, num_nodes))
        for dep in dependancys:
            adj_matrix[dep[0]][dep[1]] = dep[2]
        return adj_matrix

    def matrix_to_CSR(self, matrix):
        '''
        Retursn matrix in compressed sparse row format
        '''
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
        '''
        Creates a custom text based representation of the node list
        '''
        with open('custon_graph.txt', "w") as f:
            for node in self.node_list:
                if node.opp != 'null':
                    f.write(f'{node.func}\n')

    def create_cost_vec(self):
        CPU_cost = [node.time[0] for node in self.node_list]
        PHU_cost = [node.time[1] for node in self.node_list]
        return CPU_cost, PHU_cost

    def get_opp_spread(self):
        '''
        Creates a distrabution of the time spent on each opporation type
        '''
        cycles_per_op = {}
        for node in self.node_list:
            cycles_per_op.setdefault(node.opp, 0)
            cycles_per_op[node.opp] += node.time[0] # CPU
        return cycles_per_op

    # Time
    def execution_time(self, hardware_selection, adj_matrix):
        '''
        optimization(srt) which hardware optimization to use from self.optimization
        '''
        total_time = 0

        # node_time
        for node in self.node_list:
            total_time += self.hardware_selection_funcs[hardware_selection](node)

        #connection time
        start, end = np.nonzero(adj_matrix)
        for i in range(len(start)):
            bits = adj_matrix[start[i]][end[i]]

            # print(f"{self.node_list[start[i]].hardware} == {self.node_list[end[i]].hardware}")
            assert self.node_list[start[i]].hardware != None
            assert self.node_list[end[i]].hardware != None
            if self.node_list[start[i]].hardware == self.node_list[end[i]].hardware: # E->E
                total_time += bits*E_E_BIT_COST
            else:
                total_time += bits*E_PH_BIT_COST

        return round(total_time, 4)

    def bit_transfer(self, node, direction = 'out'):
        '''
        Calculates the total number of bits passed out of a node
        '''
        total_bits = 0
        if direction == 'out':
            total_bits += oc.ten_elm(node.output_shapes[0])*32 #
            # for shape in node.output_shapes:
            #     total += oc.ten_elm(shape)*32 # float32 numbes
        else:
            for shape in node.input_shapes:
                total_bits += oc.ten_elm(shape)*32 # float32 numbes

        return total_bits

    def _always_CPU(self, node):
        '''
        returns the selected time and changes the node.hardwrae to reflect the choice
        node(DependancyNode)
        CPU_time(list): List of the cost of run_CPU
        PHU_time(list): list of the cost of run_PHU
        '''
        node.hardware = 'CPU'
        return node.time[0]

    def _always_PHU(self, node):
        '''
        selects the photonic hardware choice  is available and returns the time.
        changes node.hardware to reflect choice
        node(DependancyNode)
        CPU_time(list): List of the cost of run_CPU
        PHU_time(list): list of the cost of run_PHU
        '''
        CPU_time, PHU_time = node.time

        if PHU_time < np.inf:
            node.hardware = 'PHU'
            return PHU_time
        node.hardware = 'CPU'
        return CPU_time

    def _naive_min(self, node):
        '''
        selects the minimum cost hardware choice and returns the time.
        changes node.hardware to reflect choice
        node(DependancyNode)
        CPU_time(list): List of the cost of run_CPU
        PHU_time(list): list of the cost of run_PHU
        '''
        CPU_time, PHU_time = node.time

        # if (node.time[0] + self.bit_transfer(node)*E_E_BIT_COST) < (node.time[1] + self.bit_transfer(node)*E_PH_BIT_COST):

        if PHU_time < CPU_time and self.bit_transfer(node)>1_000_000:
            node.hardware = 'PHU'
            return PHU_time
        node.hardware = 'CPU'
        return CPU_time

    # scheduling
    def kahn_topo_sort(self, graph):
        '''
        produes a liner order obeying DAG
        graph(np adjancy matrix): graph to sort
        returns:
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
        returns:
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
hardware_config_list = ["run_cpu", "run_phu"]

E_PH_BIT_COST = 3 /oc.CPU_CLOCK_SPEED
E_E_BIT_COST = 1 /oc.CPU_CLOCK_SPEED #TODO get right number


# # making graph
# graph = DependancyGraph(raw_json)
# CPU_cost, PHU_cost = graph.create_cost_vec()
# adj_matrix = graph.creat_adj_matrix_node_list()
# print(len(graph.node_list))
# print(len(np.nonzero(adj_matrix)[0]))
# print(len(np.nonzero(adj_matrix)[1]))

# order, layers_list = graph.kahn_topo_sort(adj_matrix)
# working_order, working_layers_list, working_layer_count = graph.kahn_topo_sort_working(adj_matrix)

## Timing
# print(f"naive_min: {graph.execution_time('naive_min', adj_matrix)}")
# naive_min_hardware = [node.hardware for node in graph.node_list]
# print(f"always_PHU: {graph.execution_time('always_PHU', adj_matrix)}")
# always_PHU_hardware = [node.hardware for node in graph.node_list]
# print(f"always_CPU: {graph.execution_time('always_CPU', adj_matrix)}")

# for i in range(len(naive_min_hardware)):
#     print(f"{naive_min_hardware[i]} --> {always_PHU_hardware[i]}")

## Visualization
G = gv.GraphVisualization()

graph = DependancyGraph(raw_json)

print(G)
adj = graph.creat_adj_matrix_node_list()

G.visualize(layout='kk', filename='graph_group.png')
# G.visualize(layout='spring', filename='graph_group.png')
# G.visualize(layout='shell', filename='graph_group.png')
# G.visualize(layout='spectral', filename='graph_group.png')
# G.visualize(layout='planar', filename='graph_group.png')
# G.visualize(layout='circular', filename='graph_group.png')

# Plotting the adjacency matrix
# mask = adj_matrix != 0
# adj_matrix[mask] = 1
# plt.figure(figsize=(6, 6))
# plt.imshow(adj_matrix, cmap='binary', interpolation='none')
# plt.title('GPT2 Adjacency Matrix')
# plt.colorbar()
# plt.savefig('matrix.png', dpi=300)
# plt.close()
