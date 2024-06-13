import os
import json
import numpy as np
from collections import deque
import networkx as nx
from operator_calcs import opp_time_func


read_json_path = '/home/rjtomich/photonic_compiler/model_to_graph/gpt2_graph.json'
# read_json_path = '/home/rjtomich/photonic_compiler/model_to_graph/bert-base-uncased_graph.json'
# read_json_path = '/home/rjtomich/photonic_compiler/Pytorch-LeNet/simple_LeNet_graph.json'
with open(read_json_path)  as json_file:
    raw_json = json.load(json_file) # returns json file as dict

# def ten_elm(tensor_shape):
#     ans = 1
#     for dimention in tensor_shape:
#         ans *= dimention
#     return ans


# CPU, PPU
# opp_time_func = {
#     'add':      lambda i, o: ( ten_elm(o[0]), np.inf ),
#     'subtract': lambda i, o: ( ten_elm(o[0]), np.inf),
#     'multiply': lambda i, o: ( ten_elm(o[0]), np.inf),
#     'divide':   lambda i, o: (  ten_elm(o[0]), np.inf),
#     'sqrt':     lambda i, o: ( ten_elm(o[0]), np.inf),
#     'rsqrt':    lambda i, o: ( ten_elm(o[0]), np.inf),
#     'tanh':     lambda i, o: ( ten_elm(o[0]) * 4, np.inf), #e^x definition
#     'power':    lambda i, o: ( ten_elm(o[0]), np.inf),
#     'nop':      lambda i, o: ( 0, np.inf), #reshape(Non-Computational)
#     'less':     lambda i, o: ( 1, np.inf),
#     'where':    lambda i, o: ( 1, np.inf),
#     'take':     lambda i, o: ( 1, np.inf),
#     'split':    lambda i, o: ( 3, np.inf), # for each split
#     'transpose':lambda i, o: ( ten_elm(o[0]), np.inf), # unsture
#     'mean':     lambda i, o: ( (i[0][-1]+1)*i[0][-2], np.inf),
#     'softmax':  lambda i, o: ( 6*i[0][-1]*i[0][-2], np.inf),
#     'matmul':   lambda i, o: ( ten_elm(i[0])*i[1][-2]*2, np.inf),
#     'dense':    lambda i, o: ( ten_elm(i[0])*i[1][-2]*2, np.inf),
#     'pack':     lambda i, o: ( ten_elm(i[0])*i[1][-2]*2, np.inf), # another form of dense
# }


##### Woriing with JSON #####

class DependancyNode():
    def __init__(self, oppid, node, input_shapes,output_shapes):
        self.oppid = oppid
        self.func = 'null' if node['op'] == "null" else node['name']
        self.input_shapes = input_shapes
        self.output_shapes = output_shapes
        self.opp = self._find_opp(node['attrs']['func_name']) if 'attrs' in node else 'null'
        self.cycles = self._unfsed_graph_time(node)
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
        {self.cycles=}"""

    def _find_opp(self, func_name):
        name_parts = func_name.split('_')
        for part in reversed(name_parts):
            try:
                int(part)
                continue
            except:
                return part

    def _unfsed_graph_time(self, node):
        # takes node, returns clock cycles
        if 'attrs' in node:
            cycles = opp_time_func[self.opp](self.input_shapes, self.output_shapes)
            return cycles
        return 0, 0

class DependancyGraph():
    def __init__ (self, raw_json):
        self.raw_json = raw_json
        self.node_list = self.create_nodes()

    # Create every node
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
            input_shapes = [ajusted_shapes[shape_idx[0]] for shape_idx in node['inputs']]
            output_shapes = [ajusted_shapes[index] for i in range(num_output)]
            nodes.append(DependancyNode(oppid, node, input_shapes, output_shapes))
        return nodes

    # Create the Adjancy Matrix for dependancy (from CSR format)
    def creat_dependancy(self):
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
        CPU_cost = np.full(shape=(len(self.node_list)), fill_value=np.inf)
        P_cost = np.full(shape=(len(self.node_list)), fill_value=np.inf)
        for index, node in enumerate(self.node_list):
            print(node.cycles)
            P_cost[index] = node.cycles[1]
            CPU_cost[index] = node.cycles[0]
        return (CPU_cost, P_cost)

    def total_time(self):
        total = 0
        for node in graph.node_list:
            total += node.cycles[0]
        clock_speed = 3.5 * 10**9
        ELECTRONIC_TIME_MULTIPLIER = 1/clock_speed
        print(f"{total} cycles")
        print(f"{total*ELECTRONIC_TIME_MULTIPLIER} s")
        print(f"{total*ELECTRONIC_TIME_MULTIPLIER*1000} ms")


# Constants
photonic_opps = ['dense', 'matmul']


graph = DependancyGraph(raw_json)
CPU_cost, P_cost = graph.create_cost_vec()
dep_graph = graph.creat_dependancy()
print(f"{len(CPU_cost)=}")
print(f"{len(P_cost)=}")
print(f"{len(graph.node_list)=}")
print(f"{len(graph.raw_json['nodes'])=}")
print(f"{dep_graph.shape=}")

data, column_offset, node_row_ptr = graph. matrix_to_CSR(dep_graph)
print(f"{len(data)=}")
print(f"{len(column_offset)=}")
print(f"{len(node_row_ptr)=}")
print(f"{len(graph.raw_json['node_row_ptr'])=}")

graph.total_time()

# for node in graph.node_list:
#     if node.opp == 'mean':
#         print(node.input_shapes)



##### Working with script #####
