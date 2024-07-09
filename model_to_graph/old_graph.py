

class StackedNode:
    """represents a hypernode in the dependancy graph. A collection of nodes
    (or algorithms) for one operation in the graph
    """

    def __init__(
        self,
        stack_id,
        parents,
        input_shapes,
        output_shapes,
        opp=None,
        func_stack=None,
        cost_stack=None,
        relay_node=None,
    ):
        self.stack_id = stack_id
        self.opp = (
            opp
            if relay_node is None
            else (
                self._find_opp(relay_node["attrs"]["func_name"])
                if "attrs" in relay_node
                else "null"
            )
        )
        self.parents = parents
        self.input_shapes = input_shapes
        self.output_shapes = output_shapes
        self.func_stack = (
            func_stack
            if relay_node is None
            else [alg for alg, info in oc.hardware_algs.items() if self.opp == info[0]]
        )
        self.cost_stack = (
            cost_stack
            if relay_node is None
            else [
                oc.cycle_to_s(
                    oc.hardware_algs[func][3](self.input_shapes, self.output_shapes)
                )
                for func in self.func_stack
            ]
        )
        self.func_selection = 0  # func_stack and cost_stack index
        self.hardware_selection = None
        self.start_time = 0

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
            + f"{self.opp=}\n "
            + f"{self.parents=}\n "
            + f"{self.input_shapes=}\n "
            + f"{self.output_shapes=}\n "
            + f"{self.func_stack=}\n "
            + f"{self.cost_stack=}\n "
            + f"{self.func_selection=}\n "
            + f"{self.hardware_selection=}\n "
            + f"{self.start_time=}\n"
        )


class StackedGraph:
    def __init__(self, stack_list=None, raw_json=None):
        self.raw_json = raw_json
        self.stack_list = stack_list if not raw_json else self._create_nodes()
        self.id_to_idx = {v.stack_id: i for i, v in enumerate(self.stack_list)}
        self.output_nodes = self._get_output_nodes()
        self.in_nodes = {
            stack.stack_id for stack in self.stack_list if not stack.parents
        }
        self.adj_matrix = self._creat_adj_matrix()

    def __srt__(self):
        return f"{len(self.stack_list)=}"

    def get_stack(self, stack_id):
        return self.stack_list[self.id_to_idx[stack_id]]

    def _get_output_nodes(self):
        # retrives all lief nodes
        outputs = {stack.stack_id for stack in self.stack_list}
        for stack in self.stack_list:
            outputs = outputs - set(stack.parents)
        return outputs

    def _create_nodes(self):
        ajusted_shapes = []
        split_shift = 0
        for index, node in enumerate(self.raw_json["nodes"]):
            ajusted_shapes.append(
                self.raw_json["attrs"]["shape"][1][index + split_shift]
            )
            if "split" in node["name"]:
                split_shift += 2

        nodes = []
        for index, node in enumerate(self.raw_json["nodes"]):
            num_output = int(node["attrs"]["num_outputs"]) if "attrs" in node else 1
            parents = [shape_idx[0] for shape_idx in node["inputs"]]
            input_shapes = [
                ajusted_shapes[shape_idx[0]] for shape_idx in node["inputs"]
            ]
            output_shapes = [ajusted_shapes[index] for i in range(num_output)]
            nodes.append(
                StackedNode(
                    index, parents, input_shapes, output_shapes, relay_node=node
                )
            )
        return nodes

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
                total_bits += oc.ten_elm(shape) * 32  # float32 numbes

        return total_bits

    def _make_connection_matrix(self, start_node, end_node, bit_transfer):
        """Creates matrix for time(s) to connect between hardware
        start_node: start_node_id
        end_node: end_node_id
        bit_transfer: The number of bits between the hardware
        """
        start_node = self.id_to_idx[start_node]
        end_node = self.id_to_idx[end_node]

        start_stack = self.stack_list[start_node].func_stack
        end_stack = self.stack_list[end_node].func_stack
        assert type(start_stack) == type(end_stack) == list
        connection_matrix = np.empty((len(start_stack), len(end_stack)))
        for start_idx, start in enumerate(start_stack):
            for end_idx, end in enumerate(end_stack):
                start_hw = oc.hardware_algs[start][1]
                end_hw = oc.hardware_algs[end][1]
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
                        self._bit_transfer(self.stack_list[self.id_to_idx[inp]]),
                    )
                )  # (1, 2) where 1's output are 2's inputs

        # print(dependancys)

        num_nodes = len(self.stack_list)
        adj_matrix = np.empty((num_nodes, num_nodes), dtype=object)  # block matrix
        for dep in dependancys:
            start_stack = self.id_to_idx[dep[0]]
            end_stack = self.id_to_idx[dep[1]]
            connection_matrix = self._make_connection_matrix(*dep)
            adj_matrix[start_stack][end_stack] = connection_matrix
        return adj_matrix

    # pathfinding
    def make_node_matrix(self):  # TODO no?
        """returns a matrix indexed by (node,stack)"""
        nodes = []
        height = 0
        for stack in self.stack_list:
            nodes.append((self.id_to_idx[stack.stack_id], len(stack.func_stack) - 1))
            height = max(height, len(stack.func_stack))

        node_matrix = np.full((len(self.stack_list), height), np.nan)
        for node in nodes:
            for col in range(node[1] + 1):
                node_matrix[node[0]][col] = 1
        return node_matrix

    def get_stack_neighbors(self, stack_num):
        row = self.adj_matrix[stack_num]
        not_none = [i for i, v in enumerate(row) if v is not None]
        return not_none

    def _kahn_topo_sort_working(self, transpose=False, manual=None):
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
        # print(graph)

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
            if manual:
                order.append("B")  # section Break

            for _ in range(len(que)):
                par_nodes, cur_node = que.pop(0)
                for par in par_nodes:
                    node_outdegree[par] -= 1

                if not manual or cur_node not in manual:
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
                if working != "B" and node_outdegree[working] != 0:
                    if not manual or working not in manual:
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
                if (
                    len(
                        set(self.stack_list[self.id_to_idx[cut]].parents)
                        - self.in_nodes
                    )
                    > 1
                ):
                    cuts.append(cut)
        return cuts

    def get_node_groups(self, asap=True):

        if asap:  # ASAP
            order, layers_list, _ = self._kahn_topo_sort_working(transpose=False)
        else:  # ALAP
            order, _, _ = self._kahn_topo_sort_working(transpose=True)
            _, layers_list, _ = self._kahn_topo_sort_working(transpose=False)

        cuts = self._get_cuts(layers_list)

        # ignore load and store for optimization
        sparse_order = []
        for i in order:
            if i in self.in_nodes or i in self.output_nodes:
                continue
            sparse_order.append(i)

        # print(sparse_order)

        cuts = set(cuts)
        group = []
        for stack in sparse_order:
            group.append(stack)
            if stack in cuts:
                yield group
                group = [stack]
        if group:
            yield group

    # inference
    def forward(self):
        """
        time a simulated forward pass through graph using selected nodes
        """
        total_time = 0

        # stack time
        for stack in self.stack_list:
            if stack.func_selection is None:
                continue
            cost = stack.cost_stack[stack.func_selection]
            total_time += cost

        func = np.vectorize(lambda x: x is not None)
        mask = func(self.adj_matrix)
        coordinates = np.where(mask)
        coordinates = list(zip(coordinates[0], coordinates[1]))
        for in_idx, out_idx in coordinates:
            connection_matrix = self.adj_matrix[in_idx][out_idx]

            in_node_stack_idx = self.stack_list[in_idx].func_selection
            out_node_stack_idx = self.stack_list[out_idx].func_selection

            total_time += connection_matrix[in_node_stack_idx][out_node_stack_idx]

        return total_time

    def simple_schedule(self):
        order, _, _ = self._kahn_topo_sort_working(
            transpose=True, manual=self.output_nodes
        )

        # Add load_stacks JIT. ALAP
        for output in self.output_nodes:
            parent_node = self.stack_list[self.id_to_idx[output]].parents
            parent_idx = order.index(parent_node[0])
            order.insert(parent_idx + 1, output)

        # create new layers wihtout breaks 'B'
        new_order = []
        new_layers_list = []
        temp = set()
        for node in order:
            if node == "B":
                new_layers_list.append(temp)
                temp = set()
            else:
                new_order.append(node)
                temp.add(node)

        return new_order, new_layers_list

    # metrics
    def create_schedule_data(self):
        """
        returns metadata about the schedule.
        Graph must have been scheduled first.
        """
        data = {
            "hardware": [],
            "start": [],
            "end": [],
            "label": [],  # Labels for the blocks
        }

        # for stack in graph.stack_list[16:-6]:
        for stack in self.stack_list[1:]:
            data["hardware"].append(stack.hardware_selection)
            data["start"].append(stack.start_time)
            data["end"].append(
                stack.start_time + stack.cost_stack[stack.func_selection]
            )
            data["label"].append(stack.stack_id)

        df = pd.DataFrame(data)
        return df
