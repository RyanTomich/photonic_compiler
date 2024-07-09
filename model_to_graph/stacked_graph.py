import numpy as np
import pandas as pd
import operator_calcs as oc

class Node:
    """    represent one algorithm for one opperation. Created by Stack objects
    """
    def __init__(self, algorithm, stack):
        self.algorithm = algorithm
        self.stack = stack
        self.time_cost = oc.cycle_to_s(oc.hardware_algs[algorithm][3](stack.input_shapes, stack.output_shapes))
        self.power_cost = None # TODO

    def __srt__(self):
        return (
            f"{self.algorithm=}\n"
            + f"{self.stack.stack_id=}\n "
            + f"{self.input_shapes=}\n "
            + f"{self.time_cost=}\n "
            + f"{self.power_cost=}\n "
        )

    def get_algo_info(self, type):
        opp, hardware, func, cycles = oc.hardware_algs[self.algorithm]
        info = {
            'opp': opp,
            'hardware': hardware,
            'func': func,
            'cycles': cycles
        }
        return info[type]


class Stack:
    """Represents a gruop of Node objects with common in_out functionality.
    """
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
            else [Node(alg, self) for alg, info in oc.hardware_algs.items() if self.opp == info[0]]
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

class StackGraph:
    """Represents a Dependancy Graph of Stack Objects
    """
    def __init__(self, stack_list=None, raw_json=None):
        self.raw_json = raw_json
        self.stack_list = stack_list if not raw_json else self._create_stacks()
        self.id_to_idx = {v.stack_id: i for i, v in enumerate(self.stack_list)}
        self.in_stacks = {
            stack.stack_id for stack in self.stack_list if not stack.parents
        }
        self.out_stacks = self._get_out_stacks()
        self.adj_matrix = self._creat_adj_matrix()
        print('... Graph Made ...')

    def get_stack_obj(self, stack_id):
        return self.stack_list[self.id_to_idx[stack_id]]

    def _get_out_stacks(self):
        outputs = {stack.stack_id for stack in self.stack_list}
        for stack in self.stack_list:
            outputs -= set(stack.parents)
        return outputs

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
                Stack(
                    index, parents, input_shapes, output_shapes, relay_node=node
                )
            )
        return stacks

    def _bit_transfer(self, stack, direction="out"):
        """
        Calculates the total number of bits passed out of a node
        """
        total_bits = 0
        if direction == "out":
            total_bits += oc.ten_elm(
                stack.output_shapes[0]
            )  # assuming uniform outputs(splits are all the same)
        else:
            for shape in stack.input_shapes:
                total_bits += oc.ten_elm(shape)

        return total_bits * oc.BITS_PER_NUM

    def _make_connection_matrix(self, start_node, end_node, bit_transfer):
        """Creates matrix for time(s) to connect between hardware
        start_node: start_node_id
        end_node: end_node_id
        bit_transfer: The number of bits between the hardware
        """

        start_node = self.get_stack_obj(start_node)
        end_node = self.get_stack_obj(end_node)

        start_stack = start_node.node_stack
        end_stack = end_node.node_stack
        assert type(start_stack) == type(end_stack) == list

        connection_matrix = np.empty((len(start_stack), len(end_stack)))
        for start_idx, start_node in enumerate(start_stack):
            for end_idx, end_node in enumerate(end_stack):
                start_hw = start_node.get_algo_info('hardware')
                end_hw = end_node.get_algo_info('hardware')
                hw_connection = tuple(sorted((start_hw, end_hw)))
                connection_matrix[start_idx][end_idx] = oc.hw_intercon(
                    hw_connection, bit_transfer
                )

        return connection_matrix

    def _creat_adj_matrix(self):
        """
        Creates an adjancy matrix of the dependencies using stack_list
        """
        dependancys = []
        for stack in self.stack_list:
            inputs = stack.parents
            for inp in inputs:  # where each input is an index to another node.
                dependancys.append(
                    (
                        inp,
                        stack.stack_id,
                        self._bit_transfer(self.get_stack_obj(inp))
                    )
                )  # (1, 2) where 1's output are 2's inputs

        num_nodes = len(self.stack_list)
        adj_matrix = np.empty((num_nodes, num_nodes), dtype=object)  # block matrix
        for dep in dependancys:
            start_stack_idx = self.id_to_idx[dep[0]]
            end_stack_idx = self.id_to_idx[dep[1]]
            connection_matrix = self._make_connection_matrix(*dep)
            adj_matrix[start_stack_idx][end_stack_idx] = connection_matrix
        return adj_matrix
