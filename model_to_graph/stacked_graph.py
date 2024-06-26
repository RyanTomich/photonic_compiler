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

        node_matrix = np.full((len(self.stack_list), height), np.nan)
        for node in nodes:
            for col in range(node[1]+1):
                node_matrix[node[0]][col] = 1
        return node_matrix

    def get_stack_neighbors(self, stack_num):
        row = self.adj_matrix[stack_num]
        not_none = [i for i,v in enumerate(row) if v is not None]
        return not_none

    def kahn_topo_sort_working(self):
        '''
        produes a liner order obeying DAG
        graph(np adjancy matrix): graph to sort
        returns:
            order: liner working order for each node
            working_layers_list: nodes that can work on each layer
            layer_count: the amount of layers each node can work on
        '''
        graph = self.adj_matrix
        # print(graph)

        node_indegree = {}
        node_outdegree = {'START': np.inf}
        node_parents = {}
        for idx in range(len(graph)):

            node_indegree[idx] = sum([True for i in graph[:, idx] if i is not None])
            node_outdegree[idx] = sum([True for i in graph[idx, :] if i is not None])
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
                for next_node in [i for i,v in enumerate(graph[cur_node]) if v is not None]:
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

    def get_articulation_points(self):
        null = {stack.oppid for stack in self.stack_list if stack.opp == 'null'}
        liner, layer, layer_count = self.kahn_topo_sort_working()
        a = [(k-null).pop() for k in layer if len(k-null) == 1]

        return a[1:-1]
