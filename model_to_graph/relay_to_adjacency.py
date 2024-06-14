import os
import json
import numpy as np
from collections import deque
import networkx as nx
from operator_calcs import opp_time_func, ten_elm


read_json_path = '/home/rjtomich/photonic_compiler/model_to_graph/gpt2_graph.json'
# read_json_path = '/home/rjtomich/photonic_compiler/model_to_graph/bert-base-uncased_graph.json'
# read_json_path = '/home/rjtomich/photonic_compiler/Pytorch-LeNet/simple_LeNet_graph.json'
with open(read_json_path)  as json_file:
    raw_json = json.load(json_file) # returns json file as dict

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
        self.optimization = {'always_CPU': self._always_CPU, 'min': self._min}

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
            CPU_cost[index] = node.cycles[0]
            P_cost[index] = node.cycles[1]
        return (CPU_cost, P_cost)

    def get_opp_spread(self):
        cycles_per_op = {}
        for node in self.node_list:
            cycles_per_op.setdefault(node.opp, 0)
            cycles_per_op[node.opp] += node.cycles[0] # CPU
        return cycles_per_op

    # Time
    def total_time(self, optimization):
        func = self.optimization[optimization]
        total = 0
        for node in graph.node_list:
            cpu_cycles, phu_cycles = node.cycles
            total += func(node, cpu_cycles, phu_cycles)
        return(total)

    def _transfer_cost(self, node):
        floats_in = sum(ten_elm(shape) for shape in node.input_shapes)
        floats_out = sum(ten_elm(shape) for shape in node.output_shapes)
        return (floats_in + floats_out) * E_PH_COST


    def _always_CPU(self, node, cpu_cycles, phu_cycles):
        node.hardware = 'CPU'
        return cpu_cycles*CPU_time_multiplier

    def _min(self, node, cpu_cycles, phu_cycles):
        if cpu_cycles < np.inf and phu_cycles < np.inf:
            CPU_time = cpu_cycles * CPU_time_multiplier
            PHU_time = (phu_cycles * PHU_time_multiplier) + self._transfer_cost(node)

            if CPU_time < PHU_time:
                node.hardwer = 'CPU'
                return CPU_time
            else:
                node.hardwer = 'PHU'
                return PHU_time
        else:
            node.hardware = 'CPU'
            return cpu_cycles * CPU_time_multiplier

    # scheduling
    def kahn_topo_sort(self, graph):
        '''
        produes a liner order obeying DAG
        graph(np adjancy matrix): graph to sort
        order(list): liner order
        layers_list(list of lists): each entry is a dependancy layer
        '''
        node_indegree = {}
        for col in range(len(graph)):
            node_indegree[col] = np.sum(graph[:, col] != 0)

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


# Constants
photonic_opps = ['dense', 'matmul']
CPU_CLOCK_SPEED = 10**8 #.1Ghz
PHU_CLOCK_SPEED = 10**10 # 10 Ghz
CPU_time_multiplier = 1/CPU_CLOCK_SPEED
PHU_time_multiplier = 1/PHU_CLOCK_SPEED
E_PH_BIT_COST = 3 * PHU_time_multiplier
BITS_PER_FLOAT = 32
E_PH_COST = E_PH_BIT_COST * BITS_PER_FLOAT

print(CPU_time_multiplier)
print(PHU_time_multiplier)


graph = DependancyGraph(raw_json)
CPU_cost, P_cost = graph.create_cost_vec()
dep_graph = graph.creat_dependancy()
data, column_offset, node_row_ptr = graph. matrix_to_CSR(dep_graph)
order, layers = graph.kahn_topo_sort(dep_graph)

### lookin at layers
# for layer in layers:
#     if len(layer) > 1:
#         opps_in_layer = []
#         for node in layer:
#             opps_in_layer.append(graph.node_list[node].opp)
#         print (opps_in_layer)

### length validation
# print(f"{len(CPU_cost)=}")
# print(f"{len(P_cost)=}")
# print(f"{len(graph.node_list)=}")
# print(f"{len(graph.raw_json['nodes'])=}")
# print(f"{dep_graph.shape=}")

# print(f"{len(data)=}")
# print(f"{len(column_offset)=}")
# print(f"{len(node_row_ptr)=}")
# print(f"{len(graph.raw_json['node_row_ptr'])=}")

### Timing
optimization =('always_CPU', 'min')
print(f"always_CPU: {graph.total_time('always_CPU')}")
print(f"min: {graph.total_time('min')}")

print(graph.cpu_count)
print(graph.phu_count)

### misc
# print(graph.get_opp_spread())

# for node in graph.node_list:
#     if node.opp == 'dense':
#         print(node.input_shapes)



##### Working with script #####
