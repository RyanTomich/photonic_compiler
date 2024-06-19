import operator_calcs as oc
import numpy as np
import json


class StackedNode():
    def __init__(self, oppid, relay_node, parents, input_shapes, output_shapes):
        self.oppid = oppid
        self.opp = self._find_opp(relay_node['attrs']['func_name']) if 'attrs' in relay_node else 'null'
        self.parents = parents
        self.input_shapes = input_shapes
        self.output_shapes = output_shapes
        self.stack = [alg for alg in oc.hardware_algs if self.opp in alg]

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
        return f"{self.oppid} \n{self.opp} \n{self.parents}  \n{self.input_shapes} \n{self.output_shapes} \n{self.stack}"

class StackedGraph():
    def __init__(self, raw_json):
        self.raw_json = raw_json
        self.node_list = self.create_nodes()

    def create_nodes(self):
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
            nodes.append(StackedNode(index, node, parents, input_shapes, output_shapes))
        return nodes

    def bit_transfer(self, node, direction = 'out'):
        '''
        Calculates the total number of bits passed out of a node
        '''
        total_bits = 0
        if direction == 'out':
            total_bits += oc.ten_elm(node.output_shapes[0])*32 #
        else:
            for shape in node.input_shapes:
                total_bits += oc.ten_elm(shape)*32 # float32 numbes

        return total_bits

    def creat_adj_matrix_node_list(self):
        '''
        Creates an adjancy matrix of the dependencies using node_list
        '''
        dependancys = []
        for node in self.node_list:
            inputs = node.parents
            for inp in inputs: # where each input is an index to another node.
                dependancys.append( (inp, node.oppid, self.bit_transfer(self.node_list[inp])) ) # (1, 2) where 1's output are 2's inputs


        num_nodes = (len(self.node_list))
        # adj_matrix = [] # block matrix
        adj_matrix = np.empty((num_nodes, num_nodes), dtype=object) # block matrix
        for dep in dependancys:
            connection_matrix = self.make_connection_matrix(dep[0], dep[1])
            adj_matrix[dep[0]][dep[1]] = (dep[2], connection_matrix)
        return adj_matrix

    def make_connection_matrix(self, start_node, end_node):
        '''
        start_node: start_node_id
        end_node: end_node_id
        '''
        start_stack = self.node_list[start_node].stack
        end_stack = self.node_list[end_node].stack
        assert type(start_stack) == type(end_stack) == list
        connection_matrix = np.empty((len(start_stack) ,len(end_stack) ))
        # TODO fill in connection cost
        return connection_matrix


read_json_path = '/home/rjtomich/photonic_compiler/model_to_graph/gpt2_graph.json'
# read_json_path = '/home/rjtomich/photonic_compiler/model_to_graph/bert-base-uncased_graph.json'
# read_json_path = '/home/rjtomich/photonic_compiler/Pytorch-LeNet/simple_LeNet_graph.json'
with open(read_json_path)  as json_file:
    raw_json = json.load(json_file) # returns json file as dict


# node = StackedNode(0, json_node, [1,2], (), ())
# print(node)


graph = StackedGraph(raw_json)
adj_matrix = graph.creat_adj_matrix_node_list()
