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

def tensor_elements(tensor_shape):
    ans = 1
    for dimention in tensor_shape:
        ans *= dimention
    return ans


# max(out, key=len)

opp_time_func = {
    'add': lambda i, o: tensor_elements(o),
    'subtract': lambda i, o: tensor_elements(o),
    'multiply': lambda i, o: tensor_elements(o),
    'divide': lambda i, o:  tensor_elements(o),
    'sqrt': lambda i, o: tensor_elements(o),
    'rsqrt': lambda i, o: tensor_elements(o),
    'power': lambda i, o: 0,
    'mean': lambda i, o: tensor_elements(o), # + 1 division
    'nop': lambda i, o: 0, #reshape(Non-Computational)
    'less': lambda i, o: 0,
    'where': lambda i, o: 0,
    'take': lambda i, o: 0,
    'softmax': lambda i, o: 0,
    'cast': lambda i, o: 0,
    'matmul': lambda i, o: 0,
    'transpose': lambda i, o: 0,
    'split': lambda i, o: 0,
    'dense': lambda i, o: 0,
    'null': lambda i, o: 0,
    'tanh': lambda i, o: 0,
}

def find_opp(func_name):
    name_parts = func_name.split('_')
    for part in reversed(name_parts):
        try:
            int(part)
            continue
        except:
            return part

def unfsed_graph_time(node, input_shapes, output_shape):
    # takes node, returns clock cycles
    if 'attrs' in node:
        func_name = node['attrs']['func_name']
        opp = find_opp(func_name)
        cycles = opp_time_func[opp](input_shapes, output_shape)
        return cycles, opp
    return 0, 'other'


##### Woriing with JSON #####

class DependancyNode():
    def __init__(self, oppid, node, input_shapes,output_shape, hardware):
        self.oppid = oppid
        self.func = node['name']
        self.input_shapes = input_shapes
        self.output_shape = output_shape
        self.hardware = hardware
        self.time, self.opp = unfsed_graph_time(node, input_shapes, output_shape)

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
        {self.output_shape=}
        {self.hardware=}
        {self.time=}"""

# Create every node
def create_nodes(raw_json):
    nodes = []
    for index, node in enumerate(raw_json["nodes"]):
        oppid = index
        input_shapes = [raw_json['attrs']['shape'][1][shape_idx[0]] for shape_idx in node['inputs']]
        output_shape = raw_json['attrs']['shape'][1][index]
        hardware = "CPU"
        nodes.append(DependancyNode(oppid, node, input_shapes,output_shape, hardware))
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
data, column_offset, node_row_ptr = matrix_to_CSR(dependancy_matrix)

tot = 0
for n in node:
    tot += n.time

ELECTRONIC_TIME_MULTIPLIER = 10**-8
print(f"{tot*ELECTRONIC_TIME_MULTIPLIER} s")
print(f"{tot*ELECTRONIC_TIME_MULTIPLIER*1000} ms")


##### Working with script #####
