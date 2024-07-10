import numpy as np
import pandas as pd
import operator_calcs as oc


class Node:
    """represent one algorithm for one opperation. Created by Stack objects"""

    def __init__(self, algorithm, stack):
        self.algorithm = algorithm
        self.stack = stack
        self.stack_id = stack.stack_id
        self.parents = stack.parents
        self.input_shapes = stack.input_shapes
        self.output_shapes = stack.output_shapes
        self.time_cost = oc.cycle_to_s(
            oc.hardware_algs[algorithm][3](stack.input_shapes, stack.output_shapes)
        )
        self.power_cost = None  # TODO

        self.hardware_selection = None
        self.start_time = None

    def __str__(self):
        return(
            f"{self.algorithm}\n"
            + f"{self.stack_id}\n"
            + f"{self.parents}\n"
            + f"{self.input_shapes}\n"
            + f"{self.output_shapes}\n"
            + f"{self.time_cost}\n"
            + f"{self.power_cost}\n"
            + f"{self.hardware_selection}\n"
            + f"{self.start_time}\n"
        )

    def get_algo_info(self, type):
        opp, hardware, func, cycles = oc.hardware_algs[self.algorithm]
        info = {"opp": opp, "hardware": hardware, "func": func, "cycles": cycles}
        return info[type]


class Stack:
    """Represents a gruop of Node objects with common in_out functionality."""

    def __init__(
        self,
        stack_id,
        parents,
        input_shapes,
        output_shapes,
        opp=None,
        relay_node=None,
        node_stack=None,
    ):
        self.stack_id = stack_id
        self.parents = parents
        self.input_shapes = input_shapes
        self.output_shapes = output_shapes
        self.opp = (
            opp
            if relay_node is None
            else (
                self._find_opp(relay_node["attrs"]["func_name"])
                if "attrs" in relay_node
                else "null"
            )
        )
        self.node_stack = (
            node_stack
            if relay_node is None
            else [
                Node(alg, self)
                for alg, info in oc.hardware_algs.items()
                if self.opp == info[0]
            ]
        )
        self.node_selection = None

    def _find_opp(self, func_name):
        """
        func_name(srt) - whole tvm function name
        returns (srt) - last collection of letters, which is the name
        """
        name_parts = func_name.split("_")
        for part in reversed(name_parts):
            try:
                int(part)
                continue
            except ValueError:
                return part

    def __str__(self):
        return (
            f"{self.stack_id=}\n"
            + f"{self.parents=}\n "
            + f"{self.input_shapes=}\n "
            + f"{self.output_shapes=}\n "
            + f"{self.opp=}\n "
            + f"{len(self.node_stack)=}\n "
            + f"{self.node_selection=}\n "
        )


class Graph:
    def __init__(self, node_list):
        self.node_list = node_list
        self.id_to_idx = {v.stack_id: i for i, v in enumerate(self.node_list)}
        self.in_nodes = self._get_in()
        self.out_nodes = self._get_out()
        self.adj_matrix = self._creat_adj_matrix()

    def _get_in(self):
        return {node.stack_id for node in self.node_list if not node.parents}

    def _get_out(self):
        outputs = {node.stack_id for node in self.node_list}
        for node in self.node_list:
            outputs -= set(node.parents)
        return outputs

    def get_node_obj(self, stack_id):
        return self.node_list[self.id_to_idx[stack_id]]

    def get_stack_neighbors(self, idx):
        """given a stack id, return all accessable neighbors

        Args:
            stack_idx (int): stack index in StackGraph

        Returns:
            list: list of neighboring stack_idx's
        """
        row = self.adj_matrix[idx]
        not_none = [i for i, v in enumerate(row) if v is not None]
        return not_none

    # adj_matrix

    def _bit_transfer(self, node, direction="out"):
        """
        Calculates the total number of bits passed out of a node
        """
        total_bits = 0
        if direction == "out":
            total_bits += oc.ten_elm(
                node.output_shapes[0]
            )  # assuming uniform outputs(splits are all the same)
        else:
            for shape in node.input_shapes:
                total_bits += oc.ten_elm(shape)

        return total_bits * oc.BITS_PER_NUM

    def _make_connection(self, start_node, end_node, bit_transfer) -> int:
        """makes connection cost for flat graphs

        Args:
            start_node (int): start node id
            end_node (int): end node id
            bit_transfer (int): bits transfered between

        Returns:
            int: time cost between
        """
        start_node = self.get_node_obj(start_node)
        end_node = self.get_node_obj(end_node)

        start_hw = start_node.get_algo_info("hardware")
        end_hw = end_node.get_algo_info("hardware")
        hw_connection = tuple(sorted((start_hw, end_hw)))
        connection = oc.hw_intercon(hw_connection, bit_transfer)
        return connection

    def _creat_adj_matrix(self):
        """
        Creates an adjancy matrix of the dependencies using stack_list
        """
        dependancys = []
        for node in self.node_list:
            inputs = node.parents
            for inp in inputs:  # where each input is an index to another node.
                dependancys.append(
                    (inp, node.stack_id, self._bit_transfer(self.get_node_obj(inp)))
                )  # (1, 2) where 1's output are 2's inputs

        num_nodes = len(self.node_list)
        adj_matrix = np.empty((num_nodes, num_nodes), dtype=object)  # block matrix
        for dep in dependancys:
            start_stack_idx = self.id_to_idx[dep[0]]
            end_stack_idx = self.id_to_idx[dep[1]]
            connection = self._make_connection(*dep)
            adj_matrix[start_stack_idx][end_stack_idx] = connection
        return adj_matrix


class StackGraph(Graph):
    """Represents a Dependancy Graph of Stack Objects"""

    def __init__(self, stack_list=None, raw_json=None):
        self.raw_json = raw_json
        self.stack_list = stack_list if not raw_json else self._create_stacks()
        super().__init__(self.stack_list)
        assert self.node_list == self.stack_list
        print("... Graph Made ...")

    def _create_stacks(self):
        ajusted_shapes = []
        split_shift = 0
        for index, node in enumerate(self.raw_json["nodes"]):
            ajusted_shapes.append(
                self.raw_json["attrs"]["shape"][1][index + split_shift]
            )
            if "split" in node["name"]:
                split_shift += 2

        stacks = []
        for index, node in enumerate(self.raw_json["nodes"]):
            num_output = int(node["attrs"]["num_outputs"]) if "attrs" in node else 1
            parents = [shape_idx[0] for shape_idx in node["inputs"]]
            input_shapes = [
                ajusted_shapes[shape_idx[0]] for shape_idx in node["inputs"]
            ]
            output_shapes = [ajusted_shapes[index] for i in range(num_output)]
            stacks.append(
                Stack(index, parents, input_shapes, output_shapes, relay_node=node)
            )

        oc.NODE_COUNT = max(oc.NODE_COUNT, index)
        return stacks

    # adj_matrix

    def _make_connection(self, start_node, end_node, bit_transfer) -> np.array:
        """makes connection cost for flat graphs

        Args:
            start_node (int): start node id
            end_node (int): end node id
            bit_transfer (int): bits transfered between

        Returns:
            matrix: time cost between all nodes in the two stacks
        """

        start_node = self.get_node_obj(start_node)
        end_node = self.get_node_obj(end_node)

        start_stack = start_node.node_stack
        end_stack = end_node.node_stack
        assert type(start_stack) == type(end_stack) == list

        connection_matrix = np.empty((len(start_stack), len(end_stack)))
        for start_idx, start_node in enumerate(start_stack):
            for end_idx, end_node in enumerate(end_stack):
                start_hw = start_node.get_algo_info("hardware")
                end_hw = end_node.get_algo_info("hardware")
                hw_connection = tuple(sorted((start_hw, end_hw)))
                connection_matrix[start_idx][end_idx] = oc.hw_intercon(
                    hw_connection, bit_transfer
                )

        return connection_matrix

    # Node_selection

    def _kahn_topo_sort_working(self, transpose=False):
        """
        Reversed True = ALAP (as late as posiable)
        Reversed False = ASAP (as soon as posiable)

        produes a liner order obeying DAG
        graph(np adjancy matrix): graph to sort
        returns:
            order: liner working order for each node
            working_layers_list: nodes whos results are still needed
            layer_count: number of layers before the stackes results are no longe needed.
        """
        graph = self.adj_matrix

        if transpose:
            graph = graph.T

        node_indegree = {}
        node_outdegree = {"START": np.inf}
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
                que.append((["START"], node))
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
                for next_node in [
                    i for i, v in enumerate(graph[cur_node]) if v is not None
                ]:
                    node_indegree[next_node] -= 1
                    node_parents[next_node].append(cur_node)
                    if node_indegree[next_node] == 0:
                        que.append((node_parents[next_node], next_node))
                        layer_count[next_node] = 0

            for working in order:
                if node_outdegree[working] != 0:
                    layers_dic[layer].add(working)
                    layer_count[working] += 1

        assert any(node_indegree.values()) is False

        layers_list = [val for (key, val) in layers_dic.items()]
        if transpose:
            return list(reversed(order)), list(reversed(layers_list)), layer_count
        return order, layers_list, layer_count

    def _get_cuts(self, layers_list):
        """returns the bridging nodes in the graph using topological sort
        layers_list: a list of layers of stacks whos results are still needed
        """
        cuts = []
        for layer in layers_list:
            if len(layer - self.in_nodes) == 1:
                cut = (layer - self.in_nodes).pop()
                if len(set(self.get_node_obj(cut).parents) - self.in_nodes) > 1:
                    cuts.append(cut)
        return cuts

    def get_node_groups(self, asap=True):
        """generates groups cut at Articulation points

        Args:
            asap (bool, optional): nodes places as soon as dependancies are done. Defaults to True.
            false means alap, nodes are placed after dependancies, but right before first child
        Yields:
            list: list of stack_id's
        """
        if asap:
            order, layers_list, _ = self._kahn_topo_sort_working(transpose=False)
        # ALAP
        else:
            order, _, _ = self._kahn_topo_sort_working(transpose=True)
            _, layers_list, _ = self._kahn_topo_sort_working(transpose=False)

        cuts = self._get_cuts(layers_list)

        # ignore load and store for optimization
        sparse_order = []
        for i in order:
            if i not in self.in_nodes and i not in self.out_nodes:
                sparse_order.append(i)

        cuts = set(cuts)
        group = []
        for stack in sparse_order:
            group.append(stack)
            if stack in cuts:
                yield group
                group = [stack]
        if group:
            yield group
