import os
import json
import numpy as np
from collections import deque
import networkx as nx



read_json_path = '/home/rjtomich/photonic_compiler/model_to_graph/gpt2_graph.json'
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
nodes = []
for index, node in enumerate(raw_json["nodes"]):
    oppid = index
    # func = node['attrs']['func_name'] if 'attrs' in
    func = node['name']
    opp = None
    input_size = [raw_json['attrs']['shape'][1][input_index[0]] for input_index in node['inputs']]
    hardware = "CPU"
    nodes.append(DependancyNode(oppid, func, opp, input_size, hardware))

# Create the Adjancy Matrix for dependancy (from CSR format)
def csr_to_matrix(csr):
    print(type(csr))
    adj_matrix = np.empty((len(csr), max(csr)+1))
    print(adj_matrix.shape)
    for index, val in enumerate(csr): # index is row, val is column
        adj_matrix[index][val] = True
    return adj_matrix

print(np.count_nonzero(csr_to_matrix(raw_json["node_row_ptr"])))
