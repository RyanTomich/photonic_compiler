import copy
import heapq
import numpy as np
import stacked_graph as sg

import testing as test
import operator_calcs as oc

# region ###### Rolling Dijkstra for embeded branched stacked graphs ######


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
    for stack_id in graph.load_nodes:
        que.append((0, ((graph.id_to_idx[stack_id], 0),)))

    groups = (
        []
    )

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
            for node, node_cost in enumerate(graph.stack_list[neighbor].cost_stack):
                edge_weight = stack_connection[cur_node[1]][node]
                new_distance = cur_dist + node_cost + edge_weight
                heapq.heappush(que, (new_distance, cur_path + ((neighbor, node),)))


def graph_partition(graph):
    """Finds the Articulation Vertices and partitions the large graph into subgraphs
    StackedGraph objects. Inclusive on both ends of range.
    graph: StackedGraph
    """
    groups = list(graph.get_node_groups(asap=False))
    assert test.group_validate(graph, groups)

    for group in groups:

        start_stack = sg.StackedNode(
            0, [], [[]], [[]], opp="start", func_stack=["start"], cost_stack=[0]
        )
        first_stack = copy.deepcopy(graph.stack_list[graph.id_to_idx[group[0]]])
        first_stack.parents = [0]

        subgraph_stack_list = [start_stack, first_stack]
        for stack_id in group[1:]:
            stack = graph.stack_list[graph.id_to_idx[stack_id]]
            new_node = copy.deepcopy(stack)
            new_node.parents = set(new_node.parents) - graph.load_nodes
            subgraph_stack_list.append(new_node)

        sub_graph = sg.StackedGraph(stack_list=subgraph_stack_list)
        yield sub_graph


def select_nodes(graph, subgraphs):
    """apply roling_dijkstra to each subgraph. Then apply those selections to the nodes
    of the original graph.
    graph: StackedGraph
    subgraph: StackedGraph
    such that subgraph is a partition of graph
    """
    for subgraph in subgraphs:
        nodes = rolling_dijkstra(subgraph)
        for node in nodes:
            stack_stack_id = subgraph.stack_list[node[0]].stack_id

            subgraph_stack = subgraph.stack_list[subgraph.id_to_idx[stack_stack_id]]
            subgraph_stack.func_selection = node[1]

            original_stack = graph.stack_list[graph.id_to_idx[stack_stack_id]]
            original_stack.func_selection = node[1]


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
    visited = {0}
    end_times = {
        0: 0
    }  # TODO handling multiple input nodes not all point to 0 Like for BERT graph 0
    indegree = {idx: len(stack.parents) for idx, stack in enumerate(graph.stack_list)}
    que = []
    for stack_id in graph.load_nodes:
        que.append((graph.id_to_idx[stack_id],))

    while que:
        # select the one that can be done the soonest, parents have the earlies end time.
        small_val = np.inf
        small_idx = np.inf
        for idx, v in enumerate(que):
            if end_times[graph.stack_list[v[-1]].stack_id] < small_val:
                small_val = end_times[graph.stack_list[v[-1]].stack_id]
                small_idx = idx

        cur_path = que.pop(small_idx)
        cur_node = cur_path[-1]

        neighbor_stacks = graph.get_stack_neighbors(cur_node)

        for neighbor in neighbor_stacks:
            indegree[neighbor] -= 1
            if neighbor not in visited and indegree[neighbor] == 0:
                neighbor_node = graph.stack_list[neighbor]
                hardware_type = oc.hardware_algs[
                    neighbor_node.func_stack[neighbor_node.func_selection]
                ][1]

                parent_end = [end_times[parent] for parent in neighbor_node.parents]
                parent_time = max(parent_end)

                # select hardware
                less_than = [
                    available
                    for available in oc.available_hardware[hardware_type]
                    if oc.available_hardware[hardware_type][available] <= parent_time
                ]

                if not less_than:
                    selected_hardware = min(
                        oc.available_hardware[hardware_type],
                        key=lambda k: oc.available_hardware[hardware_type][k],
                    )  # just use smallest
                else:
                    selected_hardware = min(
                        less_than,
                        key=lambda k: parent_time
                        - oc.available_hardware[hardware_type][k],
                    )  # select minimum distance away
                    oc.available_hardware[hardware_type][
                        selected_hardware
                    ] = parent_time  # realign hardware


                neighbor_node.hardware_selection = selected_hardware
                assert neighbor_node.start_time == 0
                neighbor_node.start_time = oc.available_hardware[hardware_type][
                    selected_hardware
                ]

                # add time
                stack_connection = graph.adj_matrix[cur_node][neighbor]

                node_cost = neighbor_node.cost_stack[neighbor_node.func_selection]
                edge_weight = stack_connection[
                    graph.stack_list[cur_node].func_selection
                ][neighbor_node.func_selection]
                oc.available_hardware[hardware_type][selected_hardware] += (
                    node_cost + edge_weight
                )
                # oc.available_hardware[hardware_type][selected_hardware] += node_cost
                new_time = oc.available_hardware[hardware_type][selected_hardware]
                visited.add(neighbor)
                end_times[neighbor_node.stack_id] = new_time
                que.append(cur_path + (neighbor,))
    return oc.available_hardware


def schdeule_nodes(graph, subgraphs):
    """
    graph = StackedGraph object, original graph
    subgraphs = Subgraph of
    """
    # merge subgraphs back to main graph
    oc.initilize_hardware()
    break_points = []
    for subgraph in subgraphs:
        hardware_times = scheduling_dijkstra(subgraph)
        for stack in subgraph.stack_list:
            original_stack = graph.stack_list[graph.id_to_idx[stack.stack_id]]
            original_stack.hardware_selection = stack.hardware_selection
            original_stack.start_time = stack.start_time
        hardware_synchronize()
        break_points.append(
            max(max(inner_dict.values()) for inner_dict in hardware_times.values())
        )

    cur_time = max(max(inner_dict.values()) for inner_dict in hardware_times.values())

    # schedule load_nodes
    adj_matrix = graph.adj_matrix

    for node in graph.load_nodes:
        stack_obj = graph.get_stack(node)
        child = [
            (idx, val) for idx, val in enumerate(adj_matrix[node]) if val is not None
        ][0]
        child_obj = graph.get_stack(child[0])
        stack_obj.start_time = (
            child_obj.start_time
            - child[1][stack_obj.func_selection][child_obj.func_selection]
            - stack_obj.cost_stack[stack_obj.func_selection]
        )
        stack_obj.hardware_selection = "memory"

    for node in graph.output_nodes:
        stack_obj = graph.get_stack(node)
        parents = stack_obj.parents
        parent_obj = graph.get_stack(parents[0])
        stack_obj.start_time = (
            parent_obj.start_time + parent_obj.cost_stack[parent_obj.func_selection]
        )
        stack_obj.hardware_selection = "memory"

    return cur_time, break_points


# endregion


# region ###### memory ######
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
    for node in graph.load_nodes:
        stack_obj = graph.get_stack(node)
        start_size += oc.ten_elm(stack_obj.output_shapes[0])
    dram_total = start_size
    dram.append((0, dram_total))

    sorted_stack_list = sorted(graph.stack_list, key=lambda x: x.start_time)

    for stack_obj in sorted_stack_list:
        in_size = sum(oc.ten_elm(x) for x in stack_obj.input_shapes)
        out_size = sum(oc.ten_elm(x) for x in stack_obj.output_shapes)
        assert out_size >= 0

        if stack_obj.stack_id in graph.load_nodes:  # D -> S
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

    print(f"{dram_total=}")
    print(f"{sram_total=}")
    return dram, sram

# endregion
