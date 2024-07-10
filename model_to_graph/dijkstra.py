import copy
import heapq
import numpy as np
import stacked_graph as sg

import testing as test
import operator_calcs as oc
import photonic_algorithms as pa

# region ###### Rolling Dijkstra for embeded branched stacked graphs ######


def graph_partition(graph):
    """Finds the Articulation Vertices and partitions the large graph into subgraphs
    StackedGraph objects. Inclusive on both ends of range.
    graph: StackedGraph
    """
    groups = list(graph.get_node_groups(asap=False))
    assert test.group_validate(graph, groups)

    for group in groups:

        start_stack = sg.Stack(0, set(), [[]], [[]], opp="start", node_stack=[])
        start_stack.node_stack.append(sg.Node("start", start_stack))

        # replace parents if not satisfied in group
        first_stacks = []
        stacks_hit = set()
        for stack in group:
            stack_obj = graph.get_node_obj(stack)
            if all(parent not in group for parent in stack_obj.parents):
                stacks_hit.add(stack_obj.stack_id)
                first_stack = copy.deepcopy(stack_obj)
                first_stack.parents = {0}
                first_stacks.append(first_stack)

        subgraph_stack_list = [start_stack] + first_stacks
        for stack_id in group:
            if stack_id not in stacks_hit:
                stack_obj = graph.get_node_obj(stack_id)
                new_node = copy.deepcopy(stack_obj)
                new_node.parents = set(new_node.parents) - graph.in_nodes
                subgraph_stack_list.append(new_node)

        sub_graph = sg.StackGraph(stack_list=subgraph_stack_list)
        yield sub_graph
    print("... Subgraphs Made ...")


def extract_stacks(path):
    # returns set of stacks included in the path
    return {index[0] for index in path}


def make_aggreement_list(graph):
    """
    graph: StackedGraph Object
    returns all stack indicies with branches or merges
    """
    branches = []
    for idx, row in enumerate(graph.adj_matrix):
        row_counts = 0
        for stack_idx, element in enumerate(row):
            if element is not None and graph.stack_list[stack_idx].opp != "null":
                row_counts += 1
        col_count = 0
        for stack_idx, element in enumerate(graph.adj_matrix[:, idx]):
            if element is not None and graph.stack_list[stack_idx].opp != "null":
                col_count += 1

        if row_counts > 1 or col_count > 1:
            branches.append(idx)

    return branches


def get_aggreement(node_indexes, aggreement_stacks):
    """
    returns tuple of the nodes in each of the aggreement stacks
    node_indexes: a path
    aggreement_stacks: list of stacks that must aggreee
    """
    if not aggreement_stacks:
        return "all"
    stack_indexes = [None for _ in aggreement_stacks]
    for node in node_indexes:
        try:
            stack_indexes[aggreement_stacks.index(node[0])] = node[1]
        except ValueError:
            continue  # non-aggreement node in path

    return tuple(stack_indexes)


def ap_works(group_ap, new_ap):
    """checks that articulation points match

    Args:
        group_ap (set): all merge points
        new_ap (set): all types

    Returns:
        bool: match
    """
    matches = True
    for idx, node in enumerate(group_ap):
        if new_ap[idx] != node and new_ap[idx] is not None and node is not None:
            matches = False
    return matches


def add_group(groups, group, stack_aggreement, cur_path, stack_coverage):
    """adds this path to the group of paths

    Args:
        groups (tuple): all groups of paths
        group (list): group to add to
        stack_aggreement (set): which nodes need to aggree
        cur_path (list): this path
        stack_coverage (set): Current path coverage
    """
    new = False
    for idx, val in enumerate(group["ap"]):
        if (val is None) != (stack_aggreement[idx] is None) and (
            val is None or stack_aggreement[idx] is None
        ):
            new = True

    if new:  # there are None's so keep original
        new_group = copy.deepcopy(group)
        new_group["ap"] = [
            (a if a is not None else b)
            for a, b in zip(new_group["ap"], stack_aggreement)
        ]
        new_group["paths"] += cur_path
        new_group["coverage_groups"].append(stack_coverage)
        new_group["total_coverage"].update(stack_coverage)
        groups.append(new_group)

    else:  # perfect match so mutate group in place
        group["ap"] = [
            (a if a is not None else b) for a, b in zip(group["ap"], stack_aggreement)
        ]
        group["paths"] += cur_path
        group["coverage_groups"].append(stack_coverage)
        group["total_coverage"].update(stack_coverage)


def rolling_dijkstra(graph):
    """
    Dijkstra untill there is full coverage on a combination of aggreement stacks
    graph to optimize
    """
    aggreement_stacks = make_aggreement_list(graph)
    # all_nodes = {i.stack_id for i in graph.stack_list if i.opp != 'null'}
    all_nodes = {i for i, v in enumerate(graph.stack_list) if v.opp != "null"}

    que = []
    for stack_id in graph.in_nodes:
        que.append((0, ((graph.id_to_idx[stack_id], 0),)))

    groups = []

    while que:
        cur_dist, cur_path = heapq.heappop(que)  # minimum
        cur_node = cur_path[-1]  # last item in path
        neighbor_stacks = graph.get_stack_neighbors(cur_node[0])

        if neighbor_stacks == []:  # ending node
            stack_aggreement = get_aggreement(cur_path, aggreement_stacks)
            stack_coverage = extract_stacks(cur_path)

            added = False
            for group in groups:
                if (
                    ap_works(group["ap"], stack_aggreement)
                    and stack_coverage not in group["coverage_groups"]
                    and group["total_coverage"] - stack_coverage != {}
                ):  # same coverage, new path
                    add_group(groups, group, stack_aggreement, cur_path, stack_coverage)
                    added = True

            if not added:
                groups.append(
                    {
                        "ap": (stack_aggreement),
                        "paths": tuple(cur_path),
                        "coverage_groups": [stack_coverage],
                        "total_coverage": stack_coverage,
                    }
                )

            for group in groups:
                # group reached full coverage:
                if group["total_coverage"] == all_nodes:
                    return set(group["paths"])

        for neighbor in neighbor_stacks:
            stack_connection = graph.adj_matrix[cur_node[0]][neighbor]
            for node, node_obj in enumerate(graph.stack_list[neighbor].node_stack):
                edge_weight = stack_connection[cur_node[1]][node]
                new_distance = cur_dist + node_obj.time_cost + edge_weight
                heapq.heappush(que, (new_distance, cur_path + ((neighbor, node),)))


def select_nodes(subgraphs):
    """apply roling_dijkstra to each subgraph.
    graph: StackedGraph
    subgraph: StackedGraph
    such that subgraph is a partition of graph
    """
    selected_node_list = []
    done = set()
    flat_subgraphs = []
    for subgraph in subgraphs:
        nodes = rolling_dijkstra(subgraph)
        subgraph_nodes_list = []
        for node in nodes:
            subgraph_stack = subgraph.stack_list[node[0]]
            subgraph_stack.node_selection = node[1]
            selected_node = subgraph_stack.node_stack[subgraph_stack.node_selection]

            # for flat subgraph
            subgraph_nodes_list.append(sg.Node(selected_node.algorithm, subgraph.get_node_obj(subgraph_stack.stack_id)))

        flat_subgraphs.append(sg.Graph(subgraph_nodes_list))

    print("... Nodes selected ...")
    return flat_subgraphs


# endregion


# region ###### scheduling_dijkstra for embeded branched stacked graphs ######
def hardware_synchronize():
    max_value = max(
        max(inner_dict.values()) for inner_dict in oc.available_hardware.values()
    )

    for sub_dict in oc.available_hardware.values():
        for key in sub_dict:
            sub_dict[key] = max_value


def scheduling_dijkstra(graph):
    """
    subgraph with mock start node.
    oc.available_hardware initilized to 0
    """
    visited = {idx for idx, val in enumerate(graph.node_list) if not val.parents}
    end_times = {val.stack_id: 0 for val in graph.node_list if not val.parents}
    indegree = {idx: len(stack.parents) for idx, stack in enumerate(graph.node_list)}
    que = []
    for stack_id in graph.in_nodes:
        que.append((graph.id_to_idx[stack_id],))

    while que:
        # select the one that can be done the soonest, parents with the earlies end time
        small_val = np.inf
        small_idx = np.inf
        for idx, v in enumerate(que):
            if end_times[graph.node_list[v[-1]].stack_id] < small_val:
                small_val = end_times[graph.node_list[v[-1]].stack_id]
                small_idx = idx

        cur_path = que.pop(small_idx)
        cur_node = cur_path[-1]

        neighbor_stacks = graph.get_stack_neighbors(cur_node)

        for neighbor in neighbor_stacks:
            indegree[neighbor] -= 1
            if neighbor not in visited and indegree[neighbor] == 0:
                neighbor_node = graph.node_list[neighbor]

                hardware_type = oc.hardware_algs[
                    neighbor_node.algorithm
                ][1]

                parent_end = [end_times[parent] for parent in neighbor_node.parents]
                max_parent_end = max(parent_end)

                # select hardware to use
                less_than = [
                    available
                    for available in oc.available_hardware[hardware_type]
                    if oc.available_hardware[hardware_type][available] <= max_parent_end
                ]

                # no hardware
                if not less_than:
                    selected_hardware = min(
                        oc.available_hardware[hardware_type],
                        key=lambda k: oc.available_hardware[hardware_type][k],
                    )
                else:
                    # select minimum distance away for tightest packing
                    selected_hardware = min(
                        less_than,
                        key=lambda k: max_parent_end
                        - oc.available_hardware[hardware_type][k],
                    )
                    # bring hardware behind hardware up to current
                    oc.available_hardware[hardware_type][
                        selected_hardware
                    ] = max_parent_end

                neighbor_node.hardware_selection = selected_hardware
                assert neighbor_node.start_time == None  # not already scheduled
                neighbor_node.start_time = oc.available_hardware[hardware_type][
                    selected_hardware
                ]

                # add time
                edge_weight = graph.adj_matrix[cur_node][neighbor]
                oc.available_hardware[hardware_type][selected_hardware] += (
                    neighbor_node.time_cost + edge_weight
                )
                new_time = oc.available_hardware[hardware_type][selected_hardware]
                visited.add(neighbor)
                end_times[neighbor_node.stack_id] = new_time
                que.append(cur_path + (neighbor,))
    return oc.available_hardware


def schdeule_nodes(graph, subgraphs): # TODO bert in-to-out issues
    """
    merges scheduled subgraph nodes back to main graph
    Schedules in and out nodes
    graph = StackedGraph object, original graph
    subgraphs = Subgraph of
    """
    # merge subgraphs back to main graph
    oc.initilize_hardware()
    break_points = []
    for subgraph in subgraphs:
        hardware_times = scheduling_dijkstra(subgraph)

        # merging subgraphs to main
        for node in subgraph.node_list:
            original_node = graph.get_node_obj(node.stack_id)
            original_node.hardware_selection = node.hardware_selection
            original_node.start_time = node.start_time

        hardware_synchronize()
        break_points.append(
            max(max(inner_dict.values()) for inner_dict in hardware_times.values())
        )

    cur_time = max(max(inner_dict.values()) for inner_dict in hardware_times.values())

    for node in graph.node_list:
        if node.stack_id not in graph.in_nodes and node.stack_id not in graph.out_nodes:
            assert node.hardware_selection is not None
            assert node.start_time is not None

    adj_matrix = graph.adj_matrix


    # schedule in_nodes
    for node in graph.in_nodes:
        node_obj = graph.get_node_obj(node)
        child = [
            (idx, cost) for idx, cost in enumerate(adj_matrix[graph.id_to_idx[node]]) if cost is not None
        ][0]
        child_obj = graph.node_list[child[0]]

        node_obj.start_time = (
            child_obj.start_time
            - child[1]
            - node_obj.time_cost
        )
        node_obj.hardware_selection = "memory"

    # schedule out_nodes
    for node in graph.out_nodes:
        node_obj = graph.get_node_obj(node)
        parents = node_obj.parents
        largest = 0
        for parent in parents:
            parent_obj = graph.get_node_obj(parent)
            largest = max(
                largest,
                parent_obj.start_time + parent_obj.time_cost
            )
        node_obj.start_time = largest
        node_obj.hardware_selection = "memory"

    print('... Nodes Schdeuled ...')
    return round(cur_time, 5), break_points


# endregion


# region ###### Node Expansion ######
def group_dot_products(m1, m2):
    """given to tensors, returns dotproducts grouped by most common vector used

    Args:
        m1 (tuple): m1 shape
        m2 (tuple): m2 shape

    Returns:
        dict: common_opperand: [unique_operands]
    """
    groups = {}
    if m1[-2] <= m2[-2]: # a <= c in axb @ bxc
        for dot_prod in pa.nd_tensor_product(m1, m2):
            groups.setdefault(dot_prod[0], (dot_prod[2],[])) [1].append(dot_prod[1])
    else: # a > c
        for dot_prod in pa.nd_tensor_product(m1, m2):
            groups.setdefault(dot_prod[1], (dot_prod[2],[])) [1].append(dot_prod[0])
    return groups



def matmul_graph(node):
    """given a photonic node, create expanded computational graph to replace it

    Args:
        node (Node): Photonic algorithm
    """
    m1, m2 = node.input_shapes
    dot_prod_groups = group_dot_products(m1, m2)

    split_node = sg.Node("split", node.stack)
    split_node.stack_id -= 0.1
    split_node.output_shapes = []

    merge_node = sg.Node("split", node.stack)
    merge_node.stack_id += 0.1
    merge_node.parents = {}
    merge_node.input_shapes = []

    node_expansion_func = pa.node_expansion[node.algorithm]
    subnodes = []
    for common_operand, operand_info in dot_prod_groups.items():
        subnodes += node_expansion_func(node, operand_info[0], common_operand, operand_info[1]) #(node, size, common, unique)
        split_node.output_shapes.append( [2,operand_info[0]] )
        merge_node.input_shapes.append( [2,operand_info[0]] )

    merge_node.parents = {subnode.stack_id for subnode in subnodes}
    return [split_node, merge_node] + subnodes


def update_children(graph, node_idx):
    node_obj = graph.node_list[node_idx]
    for child_idx in graph.get_stack_neighbors(node_idx):
        child_obj = graph.node_list[child_idx]
        child_obj.parents = [parent+0.1 if parent == node_obj.stack_id else parent for parent in child_obj.parents]


def expand_nodes(flat_subgraphs):
    """given a flat_graph, replace all photonic nodes with their complete subgraphs

    Args:
        flat_graph (Graph): entire Computation
        flat_subgraphs (Graph):
    """
    new_subgraphs = []
    for subgraph in flat_subgraphs:
        new_subgraph_node_list = []

        # add replacement nodes
        for node_idx, node in enumerate(subgraph.node_list):
            if node.algorithm in pa.node_expansion:
                replacement_nodes = matmul_graph(node)
                new_subgraph_node_list += replacement_nodes
                update_children(subgraph, node_idx)

        # add rest, some have been modified
        for node_idx, node in enumerate(subgraph.node_list):
            if node.algorithm not in pa.node_expansion:
                new_subgraph_node_list.append(node)

        new_subgraphs.append(sg.Graph(new_subgraph_node_list))
    return new_subgraphs
# endregion


# region ###### memory ###### # TODO make work with Node, Stack and Graph
def get_memory_profile(graph):
    dram = []  # (time, bits)
    dram_total = 0
    sram = []  # (time, bits)
    sram_total = 0
    outdegree = {
        idx: sum([True for i in row if i is not None])
        for idx, row in enumerate(graph.adj_matrix)
    }
    # for row in range(len(graph.adj_matrix)):
    #     outdegree[row] = sum([True for i in graph.adj_matrix[row, :] if i is not None])

    start_size = 0
    for node in graph.in_nodes:
        stack_obj = graph.get_stack(node)
        start_size += oc.ten_elm(stack_obj.output_shapes[0])
    dram_total = start_size
    dram.append((0, dram_total))

    sorted_stack_list = sorted(graph.stack_list, key=lambda x: x.start_time)

    for stack_obj in sorted_stack_list:
        in_size = sum(oc.ten_elm(x) for x in stack_obj.input_shapes)
        out_size = sum(oc.ten_elm(x) for x in stack_obj.output_shapes)
        assert out_size >= 0

        if stack_obj.stack_id in graph.in_nodes:  # D -> S
            dram_total -= out_size
            dram.append((stack_obj.start_time, dram_total))

            sram_total += out_size
            sram.append(
                (
                    stack_obj.start_time
                    + stack_obj.cost_stack[stack_obj.func_selection],
                    sram_total,
                )
            )

        elif stack_obj.stack_id in graph.output_nodes:  # S -> D
            sram_total -= in_size
            sram.append((stack_obj.start_time, sram_total))

            dram_total += out_size
            dram.append(
                (
                    stack_obj.start_time
                    + stack_obj.cost_stack[stack_obj.func_selection],
                    dram_total,
                )
            )

        else:
            for parent in stack_obj.parents:
                parent_obj = graph.get_stack(parent)
                outdegree[parent_obj.stack_id] -= 1
                # once all children are satisfied, we remove data from SRAM
                if outdegree[parent_obj.stack_id] == 0:
                    size = sum(oc.ten_elm(x) for x in parent_obj.output_shapes)
                    sram_total -= size
                    sram.append((stack_obj.start_time, sram_total))

            sram_total += out_size
            sram.append(
                (
                    stack_obj.start_time
                    + stack_obj.cost_stack[stack_obj.func_selection],
                    sram_total,
                )
            )

    # print(f"{dram_total=}")
    # print(f"{sram_total=}")
    return dram, sram


# endregion
