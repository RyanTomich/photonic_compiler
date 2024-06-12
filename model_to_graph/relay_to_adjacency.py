import os
import json
import numpy as np
from collections import deque
import networkx as nx



read_json_path = '/home/rjtomich/photonic_compiler/model_to_graph/gpt2_graph.json'
# read_json_path = '/home/rjtomich/photonic_compiler/model_to_graph/bert-base-uncased_graph.json'
# read_json_path = '/home/rjtomich/photonic_compiler/Pytorch-LeNet/simple_LeNet_graph.json'
with open(read_json_path)  as json_file:
    raw_json = json.load(json_file) # returns json file as dict

class DependancyNode():
    def __init__(self, oppid, func, opp, input_size, hardware):
        self.oppid = oppid
        self.func = func
        self.opp = opp
        self.input_size = input_size
        self.hardware = hardware
        time = self._calc_time(opp, input_size)

    def _calc_time(self, opp, inputs):
        return None
        # if opp is None:
        #     return None
        # if opp == '@':
        #     a = np.prod(inputs[0])
        #     return a*inputs[1][-1]
        # if opp in ('+-*/'):
        #     return np.prod(inputs[0])

    def __hash__(self):
        return hash(self.oppid)

    def __eq__(self, other):
        if isinstance(other, DependancyNode):
            return self.oppid == other.oppid
        return False

    def __str__(self):
        return f"{self.oppid=}\n {self.func=} \n{self.opp=} \n{self.input_size=} \n{self.hardware=}"

# Create every node
def create_nodes(raw_json):
    nodes = []
    for index, node in enumerate(raw_json["nodes"]):
        oppid = index
        # func = node['attrs']['func_name'] if 'attrs' in
        func = node['name']
        opp = None
        input_size = [raw_json['attrs']['shape'][1][input_index[0]] for input_index in node['inputs']]
        hardware = "CPU"
        nodes.append(DependancyNode(oppid, func, opp, input_size, hardware))
    return nodes

# Create the Adjancy Matrix for dependancy (from CSR format)
def creat_dependancy(raw_json):
    dependancys = []
    for node_index, node in enumerate(raw_json['nodes']):
        inputs = node.get('inputs', [])
        for inp in inputs: # where each input is an index to another node.
            dependancys.append((inp[0], node_index)) # (1, 2) where 2 takes 1's output

    num_nodes = (len(raw_json['nodes']))
    adj_matrix = np.zeros((num_nodes, num_nodes))
    for dep in dependancys:
        adj_matrix[dep[0]][dep[1]] = 1
    return adj_matrix

def matrix_to_CSR(matrix):
    data = []
    column_offset = []
    node_row_ptr = [0,]
    row_prt_total = 0
    for row_num, row in enumerate(matrix):
        for col_num, cell in enumerate(row):
            if cell:
                data.append(cell)
                column_offset.append(col_num)
                row_prt_total += 1
        node_row_ptr.append(row_prt_total)
    return data, column_offset, node_row_ptr


node = create_nodes(raw_json)
dependancy_matrix = creat_dependancy(raw_json)
#number of nodes
print(len(node))
print(len(raw_json['nodes']))
print(len(dependancy_matrix))

print(np.nonzero(dependancy_matrix[54])) # what is a dependancy for node 54

print(len(np.nonzero(dependancy_matrix)[0])) # total number of connections
data, column_offset, node_row_ptr = matrix_to_CSR(dependancy_matrix)
