"""
All complex graph operations and redefinitions
"""

import copy
import heapq
import numpy as np

import stacked_graph as sg
import testing as test
import operator_calcs as oc
import photonic_algorithms as pa

# region graph_partition


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

        sub_graph = sg.StackGraph(
            stack_list=subgraph_stack_list,
            weight_variable=graph.weight_variable,
        )
        yield sub_graph
    print("... Subgraphs Made ...") if oc.DEBUG_PRINT else None


# endregion


# region select_nodes


def _extract_stacks(path):
    """returns set of stacks included in the path

    Args:
        path (list of tuples): [ (stack, node) ]

    Returns:
        set: stacks included in the path
    """
    return {index[0] for index in path}


def _make_aggreement_list(graph):
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


def _get_aggreement(node_indexes, aggreement_stacks):
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


def _ap_works(group_ap, new_ap):
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


def _add_group(groups, group, stack_aggreement, cur_path, stack_coverage):
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


def _ending_node(cur_path, aggreement_stacks, groups, all_nodes):
    """checks if ending node has made a match. All inputs mutable

    Returns:
        set: set of working nodes if end satisfies
    """

    stack_aggreement = _get_aggreement(cur_path, aggreement_stacks)
    stack_coverage = _extract_stacks(cur_path)

    added = False
    for group in groups:
        if (
            _ap_works(group["ap"], stack_aggreement)
            and stack_coverage not in group["coverage_groups"]
            and group["total_coverage"] - stack_coverage != {}
        ):  # same coverage, new path
            _add_group(groups, group, stack_aggreement, cur_path, stack_coverage)
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

    return None


def _rolling_dijkstra(graph, weight_variable):
    """
    Dijkstra untill there is full coverage on a combination of aggreement stacks
    graph to optimize
    """
    aggreement_stacks = _make_aggreement_list(graph)
    all_nodes = {i for i, v in enumerate(graph.stack_list) if v.opp != "null"}

    que = []
    for stack_id in graph.in_nodes:
        que.append(
            (0, ((graph.id_to_idx[stack_id], 0),))
        )  # (cur_dist, ( (graph.id_to_idx[stack_id], 0), ) )
    groups = []

    while que:
        cur_dist, cur_path = heapq.heappop(que)  # minimum
        cur_node = cur_path[-1]  # last item in path
        neighbor_stacks = graph.get_stack_neighbors(cur_node[0])

        if neighbor_stacks == []:  # ending node

            found = _ending_node(cur_path, aggreement_stacks, groups, all_nodes)
            if found:
                return found

        for neighbor in neighbor_stacks:
            stack_connection = graph.adj_matrix[cur_node[0]][neighbor]
            for node, node_obj in enumerate(graph.stack_list[neighbor].node_stack):
                edge_weight = stack_connection[cur_node[1]][node]
                node_cost = oc.node_value_selection[weight_variable](node_obj)
                new_distance = cur_dist + node_cost + edge_weight
                heapq.heappush(que, (new_distance, cur_path + ((neighbor, node),)))
    return None


def select_nodes(subgraphs, weight_variable):
    """apply roling_dijkstra to each subgraph.

    Args:
        subgraphs (StackedGraph)

    Returns:
        Graph
    """
    flat_subgraphs = []
    for subgraph in subgraphs:
        nodes = _rolling_dijkstra(subgraph, weight_variable=weight_variable)
        subgraph_nodes_list = []
        for node in nodes:
            subgraph_stack = subgraph.stack_list[node[0]]
            subgraph_stack.node_selection = node[1]
            selected_node = subgraph_stack.node_stack[subgraph_stack.node_selection]

            # for flat subgraph
            subgraph_nodes_list.append(
                sg.Node(
                    selected_node.algorithm,
                    subgraph.get_node_obj(subgraph_stack.stack_id),
                )
            )

        flat_subgraphs.append(sg.Graph(subgraph_nodes_list, weight_variable = 'time')) # switch to time for scheduling
        print("...     ... Subgraph Nodes selected ...") if oc.DEBUG_PRINT else None

    print("... Nodes selected ...") if oc.DEBUG_PRINT else None
    return flat_subgraphs


# endregion


# region schedule_nodes
def _hardware_synchronize():
    """brings all hardware times up to the max"""
    max_value = max(
        max(inner_dict.values()) for inner_dict in oc.available_hardware.values()
    )

    for sub_dict in oc.available_hardware.values():
        for key in sub_dict:
            sub_dict[key] = max_value


def _scheduling_dijkstra(graph):
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
        hardware_type = None
        max_parent_end = None

        for neighbor in neighbor_stacks:
            indegree[neighbor] -= 1
            if neighbor not in visited and indegree[neighbor] == 0:
                neighbor_node = graph.node_list[neighbor]

                hardware_type = oc.hardware_algs[neighbor_node.algorithm].hardware

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
                assert neighbor_node.start_time is None  # not already scheduled
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


def _time_shift(graph, time):
    """Shift all nodes =

    Args:
        graph (Graph):
        time (int): + for forwared, - for back
    """
    for node in graph.node_list:
        node.start_time += time
        assert node.start_time >= 0, "start times not all positive"

    for sub_dict in oc.available_hardware.values():
        for key in sub_dict:
            sub_dict[key] += time


def _add_in_out(original_graph, node_list):
    """add i/o nodes to graph

    Args:
        original_graph (StackGraph): graph from Json
        node_list (list): list of nodes for new graph, scheduled
    """
    all_nodes = set()

    # input_nodes
    for node in node_list:
        all_nodes.add(node.stack_id)
        if node.algorithm != "dot_prod_phu":
            if (
                node.stack_id in original_graph.id_to_idx
                or node.stack_id + 0.1 in original_graph.id_to_idx
            ):
                node.parents = set(node.parents)
                current_parents = {int(parent) for parent in node.parents}
                parents_added = (
                    original_graph.get_node_obj(round(node.stack_id)).parents
                    - current_parents
                )
                node.parents.update(parents_added)

    for in_node in original_graph.in_nodes:
        new_node = original_graph.get_node_obj(in_node).node_stack[0]
        node_list.append(new_node)

    # output_nodes
    for out_node in original_graph.out_nodes:
        out_nod_obj = original_graph.get_node_obj(out_node)
        new_parents = []
        for parent in out_nod_obj.parents:
            if parent in all_nodes:
                new_parents.append(parent)
            else:
                new_parents.append(parent + 0.1)

        new_node = out_nod_obj.node_stack[0]
        new_node.parents = new_parents
        node_list.append(new_node)

    assert test.node_list_complete(node_list)
    print("...     ... graph i/o added ...") if oc.DEBUG_PRINT else None


def _schedule_in_out(graph):
    """Schedule the i/o nodes in a graph

    Args:
        graph (Graph)
    """
    min_start_time = np.inf
    for node in graph.in_nodes:
        node_obj = graph.get_node_obj(node)
        child = [
            (idx, cost)
            for idx, cost in enumerate(graph.adj_matrix[graph.id_to_idx[node]])
            if cost is not None
        ][0]
        child_obj = graph.node_list[child[0]]

        node_obj.start_time = child_obj.start_time - child[1] - node_obj.time_cost
        min_start_time = min(min_start_time, node_obj.start_time)
        node_obj.hardware_selection = "memory"

    # schedule out_nodes
    for node in graph.out_nodes:
        node_obj = graph.get_node_obj(node)
        parents = node_obj.parents
        largest = 0
        for parent in parents:
            parent_obj = graph.get_node_obj(parent)
            largest = max(largest, parent_obj.start_time + parent_obj.time_cost)
        node_obj.start_time = largest
        node_obj.hardware_selection = "memory"

    # 0 time pass
    if min_start_time < 0:
        _time_shift(graph, -min_start_time)

    print("...     ... graph i/o scheduled ...") if oc.DEBUG_PRINT else None


def schdeule_nodes(original_graph, subgraphs):  # TODO bert in-to-out issues
    """
    merges subgraphs
    schedules in and out nodes
    graph = StackedGraph object, original graph
    subgraphs = Subgraph of
    """
    # Schedule subgraphs and merge
    oc.initilize_hardware()

    break_points = []
    nodes_seen = set()
    full_node_list = []
    for subgraph in subgraphs:
        _scheduling_dijkstra(subgraph)

        for node in subgraph.node_list:
            if node.algorithm != "start" and node.stack_id not in nodes_seen:
                if 0 in node.parents:
                    node.parents.remove(0)
                full_node_list.append(node)
                nodes_seen.add(node.stack_id)

        _hardware_synchronize()
        break_points.append(
            max(
                max(inner_dict.values())
                for inner_dict in oc.available_hardware.values()
            )
        )
        print("...     ... Subgraph Scheduled ...") if oc.DEBUG_PRINT else None

    test.merge_i_o(full_node_list, original_graph)
    _add_in_out(original_graph, full_node_list)

    graph = sg.Graph(full_node_list, weight_variable = 'time')
    _schedule_in_out(graph)

    for node in graph.node_list:
        if node.stack_id not in graph.in_nodes and node.stack_id not in graph.out_nodes:
            assert node.hardware_selection is not None
            assert node.start_time is not None

    end_time = round(
        max(max(inner_dict.values()) for inner_dict in oc.available_hardware.values()),
        5,
    )

    print("... Nodes Schdeuled ...") if oc.DEBUG_PRINT else None
    return (
        graph,
        end_time,
        break_points,
    )


# endregion


# region expand_nodes
def _group_dot_products(m1, m2):
    """given to tensors, returns dotproducts grouped by most common vector used

    Args:
        m1 (tuple): m1 shape
        m2 (tuple): m2 shape

    Returns:
        dict: common_opperand: [unique_operands]
    """
    groups = {}
    if m1[-2] <= m2[-2]:  # a <= c in axb @ bxc
        for dot_prod in pa.nd_tensor_product(m1, m2):
            groups.setdefault(dot_prod[0], (dot_prod[2], []))[1].append(dot_prod[1])
    else:  # a > c
        for dot_prod in pa.nd_tensor_product(m1, m2):
            groups.setdefault(dot_prod[1], (dot_prod[2], []))[1].append(dot_prod[0])
    return groups


def _matmul_graph(node):
    """given a photonic node, create expanded computational graph to replace it

    Args:
        node (Node): Photonic algorithm
    """
    m1, m2 = node.input_shapes
    dot_prod_groups = _group_dot_products(m1, m2)

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
        size, unique_operands = operand_info
        subnodes += node_expansion_func(
            node, size, common_operand, unique_operands
        )  # (node, size, common_operand, unique_operands)

    for subnode in subnodes:
        split_node.output_shapes += subnode.input_shapes
        merge_node.input_shapes += subnode.output_shapes

    merge_node.parents = {subnode.stack_id for subnode in subnodes}
    return [split_node, merge_node] + subnodes


def _update_children(graph, node_idx):
    """propogate patrent id change to children

    Args:
        graph (Graph):
        node_idx (int): location of node that changed
    """
    node_obj = graph.node_list[node_idx]
    for child_idx in graph.get_stack_neighbors(node_idx):
        child_obj = graph.node_list[child_idx]
        child_obj.parents = [
            parent + 0.1 if parent == node_obj.stack_id else parent
            for parent in child_obj.parents
        ]


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
                replacement_nodes = _matmul_graph(node)
                new_subgraph_node_list += replacement_nodes
                _update_children(subgraph, node_idx)

        # add rest, some have been modified
        for node_idx, node in enumerate(subgraph.node_list):
            if node.algorithm not in pa.node_expansion:
                new_subgraph_node_list.append(node)

        new_subgraphs.append(
            sg.Graph(new_subgraph_node_list, subgraph.weight_variable)
        )
        print("...     ... sungraph Nodes Expanded ...") if oc.DEBUG_PRINT else None

    print("... Nodes Expanded ...") if oc.DEBUG_PRINT else None
    return new_subgraphs


# endregion
