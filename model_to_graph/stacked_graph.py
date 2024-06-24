import operator_calcs as oc
import numpy as np
import json


class StackedNode():
    def __init__(self, oppid, parents, input_shapes, output_shapes, opp=None, func_stack=None, cost_stack=None, relay_node=None):
        self.oppid = oppid
        self.opp = opp if relay_node is None else (self._find_opp(relay_node['attrs']['func_name']) if 'attrs' in relay_node else 'null')
        self.parents = parents
        self.input_shapes = input_shapes
        self.output_shapes = output_shapes
        self.func_stack = func_stack if relay_node is None else [alg for alg in oc.hardware_algs if self.opp == oc.hardware_algs[alg][0]]
        self.cost_stack = cost_stack if relay_node is None else [oc.hardware_algs[func][3](self.input_shapes, self.output_shapes) for func in self.func_stack]


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

    def __str__(self):
        return f"{self.oppid=} \n {self.opp=} \n {self.parents=} \n {self.input_shapes=} \n {self.output_shapes=} \n {self.func_stack=} \n {self.cost_stack=} \n"

class StackedGraph():
    def __init__(self, stack_list=None, raw_json=None):
        self.raw_json = raw_json
        self.stack_list = stack_list if not raw_json else self._create_nodes()
        self.adj_matrix = self._creat_adj_matrix()

    def _create_nodes(self):
        ajusted_shapes = []
        split_shift = 0
        for index, node in enumerate(self.raw_json["nodes"]):
            ajusted_shapes.append(self.raw_json['attrs']['shape'][1][index + split_shift])
            if 'split' in node['name']:
                split_shift += 2

        nodes = []
        for index, node in enumerate(self.raw_json["nodes"]):
            num_output = int(node['attrs']['num_outputs']) if 'attrs' in node else 1
            parents = [shape_idx[0] for shape_idx in node['inputs']]
            input_shapes = [ajusted_shapes[shape_idx[0]] for shape_idx in node['inputs']]
            output_shapes = [ajusted_shapes[index] for i in range(num_output)]
            nodes.append(StackedNode(index, parents, input_shapes, output_shapes, relay_node=node))
        return nodes

    # adj_matrix
    def _bit_transfer(self, node, direction = 'out'):
        '''
        Calculates the total number of bits passed out of a node
        '''
        total_bits = 0
        if direction == 'out':
            total_bits += oc.ten_elm(node.output_shapes[0]) #assuming uniform outputs(splits are all the same)
        else:
            for shape in node.input_shapes:
                total_bits += oc.ten_elm(shape)*32 # float32 numbes

        return total_bits

    def _make_connection_matrix(self, start_node, end_node, bit_transfer):
        '''
        start_node: start_node_id
        end_node: end_node_id
        '''
        start_stack = self.stack_list[start_node].func_stack
        end_stack = self.stack_list[end_node].func_stack
        assert type(start_stack) == type(end_stack) == list
        connection_matrix = np.empty(( len(start_stack) ,len(end_stack) ))
        for start_idx in range(len(start_stack)):
            for end_idx in range(len(end_stack)):
                start_hw = oc.hardware_algs[start_stack[start_idx]][1]
                end_hw = oc.hardware_algs[end_stack[end_idx]][1]
                hw_connection = tuple(sorted( (start_hw, end_hw) ))
                connection_matrix[start_idx][end_idx] = oc.hw_intercon[hw_connection](bit_transfer)

        return connection_matrix

    def _creat_adj_matrix(self):
        '''
        Creates an adjancy matrix of the dependencies using stack_list
        '''
        dependancys = []
        for stack in self.stack_list:
            inputs = stack.parents
            for inp in inputs: # where each input is an index to another node.
                dependancys.append( (inp, stack.oppid, self._bit_transfer(self.stack_list[inp])) ) # (1, 2) where 1's output are 2's inputs


        num_nodes = (len(self.stack_list))
        adj_matrix = np.empty((num_nodes, num_nodes), dtype=object) # block matrix
        for dep in dependancys:
            connection_matrix = self._make_connection_matrix(*dep)
            adj_matrix[dep[0]][dep[1]] = connection_matrix
        return adj_matrix

    def make_node_matrix(self):
        '''returns a matrix indexed by (node,stack)'''
        nodes = []
        height = 0
        for stack in self.stack_list:
            nodes.append((stack.oppid, len(stack.func_stack)-1))
            height = max(height, len(stack.func_stack))

        print(nodes)
        node_matrix = np.full((len(self.stack_list), height), np.nan)
        for node in nodes:
            for col in range(node[1]+1):
                node_matrix[node[0]][col] = 1
        return node_matrix

    def connected(position1,position2):
        '''positions from node_matrix'''






# read_json_path = '/home/rjtomich/photonic_compiler/model_to_graph/gpt2_graph.json'
# read_json_path = '/home/rjtomich/photonic_compiler/model_to_graph/bert-base-uncased_graph.json'
# read_json_path = '/home/rjtomich/photonic_compiler/Pytorch-LeNet/simple_LeNet_graph.json'
# with open(read_json_path)  as json_file:
#     raw_json = json.load(json_file) # returns json file as dict


# g = StackedGraph(raw_json)
# for node in g.stack_list:
#     print(node.cost_stack)


# import time
# import sys
# start_time = time.time()
# g = StackedGraph(raw_json)
# end_time = time.time()
# size = sys.getsizeof(g)

# 0.010859012603 - 11684107
# 0.011414766311 - 13784141
# 0.000073671340 - 18891
