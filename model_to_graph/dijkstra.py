import numpy as np
import heapq
import stacked_graph as sg
import copy

import testing as test
import graph_visualization as gv
import operator_calcs as oc

#region ###### General Dijkstra for normal graph ######
# dependancies = [
#     (0, 1, 1), (0, 2, 2), (0,3,2),

#     (1,4, 2), (1,5,1), (1,6,2),
#     (2,4,2), (2,5,2), (2,6,2),
#     (3,4,2), (3,5,2), (3,6,2),

#     (4,7,2), (5,7,1), (6,7,2),
# ]

# values = [1, 10, 100, 100, 100, 10, 100, 10]
# # make adj matrix
# adj_matrix = np.zeros((len(values), len(values)))
# for dep in dependancies:
#     adj_matrix[dep[0]][dep[1]] = dep[2]

# # print(adj_matrix)

# def matrix_dijkstra(adj_matrix, node_weights, start = 0):
#     n = len(node_weights)
#     dist = [np.inf] * n
#     dist[start] = node_weights[start]
#     visited = [False]*n
#     que = [(node_weights[start], start)] # (distance, node)
#     previous = [-1] * n

#     while que:
#         cur_dist, u = heapq.heappop(que)

#         if visited[u]:
#             continue

#         visited[u] = True

#         for v in range(n):
#             if adj_matrix[u][v] != 0 and not visited[v]:
#                 new_dist = cur_dist + adj_matrix[u][v] + node_weights[v]

#                 if new_dist < dist[v]:
#                     dist[v] = new_dist
#                     heapq.heappush(que, (new_dist, v))
#                     previous[v] = u

#     return dist, previous



# def get_path(previous, target):
#     path = []
#     while target != -1:
#         path.insert(0, target)
#         target = previous[target]
#     return path

# dist, previous = matrix_dijkstra(adj_matrix, values, start = 0)
# print(dist)
# print(previous)
# print(get_path(previous, len(adj_matrix)-1))

#endregion

# region ###### Special Dijkstra for liner stacked graphs ######

# def stacked_dijkstra(graph, start):
#     node_matrix = graph.make_node_matrix()

#     dist = np.copy(node_matrix)
#     dist[dist == 1] = np.inf
#     dist[start] = graph.stack_list[start[0]].cost_stack[start[1]]

#     visited = np.copy(node_matrix)
#     visited[visited == 1] = 0

#     previous = np.full(node_matrix.shape, np.nan, dtype=object)
#     non_zero = np.argwhere(node_matrix == 1)
#     for idx in non_zero:
#         previous[tuple(idx)] = (-1, -1)

#     que = [(dist[start], start)] # (distance, node)

#     while que:
#         cur_dist, cur_position = heapq.heappop(que)
#         if visited[cur_position]:
#             continue

#         visited[cur_position] = True

#         for board_position in np.ndindex(node_matrix.shape):
#             # if not in visited and their stacks the nodes stacks are connected
#             stack_connection = graph.adj_matrix[cur_position[0]][board_position[0]]
#             if visited[board_position] == 0 and stack_connection is not None:
#                 next_node_weight = graph.stack_list[board_position[0]].cost_stack[board_position[1]]
#                 edge_weight = stack_connection[cur_position[1]][board_position[1]]
#                 new_dist = cur_dist + edge_weight + next_node_weight

#                 if new_dist < dist[board_position]:
#                     dist[board_position] = new_dist
#                     heapq.heappush(que, (new_dist, board_position))
#                     previous[board_position] = cur_position

#     return dist, previous

# def stacked_get_path(previous, target):
#     print(target)
#     path = []
#     while target != (-1, -1):
#         path.insert(0, target)
#         target = previous[target]
#     return path

#endregion

# region ###### Dijkstra for branched stacked graphs ######

# def get_combinations(graph, aggreement_stacks):
#     '''
#     returns all the combinations of nodes within the agreed stacks
#     graph: StackedGraph
#     aggreement_stacks: list of stacks that must aggree
#     '''
#     def combinations(nums):
#         if len(nums) == 1:
#             return [(i,) for i in range(nums[0])]
#         ans = []
#         for i in range(nums[0]):
#             for j in combinations(nums[1:]):
#                 ans.append( (i,) + j)

#         return ans

#     if aggreement_stacks:
#         return combinations([len(graph.stack_list[stack].func_stack) for stack in aggreement_stacks])
#     return ['all']

# def get_aggreement(node_indexes, aggreement_stacks):
#     '''
#     returns tuple of the nodes in each of the aggreement stacks
#     node_indexes: a path
#     aggreement_stacks: list of stacks that must aggreee
#     '''
#     if not aggreement_stacks:
#         return 'all'
#     stack_indexes = [None for _ in aggreement_stacks]
#     for node in node_indexes:
#         try:
#             stack_indexes[aggreement_stacks.index(node[0])] = node[1]
#         except ValueError:
#             continue # non-aggreement node in path

#     return tuple(stack_indexes)

# def extract_stacks(path):
#     # returns set of stacks included in the path
#     return {index[0] for index in path}

# def merge_paths(paths):
#     '''
#     given a sorted by time list of paths with full coverage, returns all nodes in
#     the optimal path.
#     [(distance, [path]), (distance, [path]), (distance, [path])]
#     '''
#     row_coverage = set()
#     nodes = []
#     for path in paths:
#         for node in path[1]:
#             if node[0] not in row_coverage:
#                 nodes.append(node)
#                 row_coverage.add(node[0])
#     return sorted(list(nodes), key=lambda x: x[0])

# def make_aggreement_list(graph):
#     '''
#     graph: StackedGraph Object
#     returns all stack indicies with branches or merges
#     '''
#     branches = []
#     for idx, row in enumerate (graph.adj_matrix):
#         row_counts = 0
#         for stack_idx, element in enumerate(row):
#             if element is not None and graph.stack_list[stack_idx].opp != 'null':
#                 row_counts += 1
#         col_count = 0
#         for stack_idx, element in enumerate(graph.adj_matrix[:, idx]):
#             if element is not None and graph.stack_list[stack_idx].opp != 'null':
#                 col_count += 1

#         if row_counts > 1 or col_count > 1:
#             branches.append(idx)

#     return(branches)

# def branching_stacked_dijkstra(graph, start=(0,0)):
#     '''
#     Returns optimal path covering all stacks
#     graph: StackedGraph object
#     start: starting node (stack, node)
#     '''
#     aggreement_stacks = make_aggreement_list(graph)
#     print(f'{aggreement_stacks=}')
#     print(f'{get_combinations(graph, aggreement_stacks)=}')

#     que = [(0, [start]) ] #(distance, [path])
#     paths = {x : [] for x in get_combinations(graph, aggreement_stacks)}
#     coverage = {x : {i.oppid for i in graph.stack_list if i.opp != 'null'} for x in get_combinations(graph, aggreement_stacks)}

#     while que:
#         print(que)
#         cur_dist, cur_path = heapq.heappop(que) # minimum
#         cur_node = cur_path[-1] # last item in path
#         neighbor = graph.get_stack_neighbors(cur_node[0])
#         if neighbor == []: # ending node
#             aggreement = get_aggreement(cur_path, aggreement_stacks)
#             print(aggreement)
#             # Paths come in ordered, so only keep paths that add new stacks.

#             if extract_stacks(cur_path) & coverage[aggreement] != {}: # There are unique stacks in current path
#                 paths[aggreement].append((cur_dist, cur_path))
#                 coverage[aggreement] -= extract_stacks(cur_path)

#             if not coverage[aggreement]: # That combination of node stack aggreement has full coverage.
#                 return merge_paths(paths[aggreement])

#         for negibor_stack in neighbor:
#             stack_connection = graph.adj_matrix[cur_node[0]][negibor_stack]
#             for node, node_cost in enumerate(graph.stack_list[negibor_stack].cost_stack):
#                 edge_weight = stack_connection[cur_node[1]][node]
#                 node_cost = sum(node_cost.values()) if isinstance(node_cost, dict) else node_cost
#                 # new_distance = cur_dist + sum(node_cost.values()) + edge_weight
#                 new_distance = cur_dist + node_cost + edge_weight
#                 heapq.heappush(que, (new_distance, cur_path + [(negibor_stack, node)]))

# endregion

# region ###### Rolling Dijkstra for embeded branched stacked graphs ######

def extract_stacks(path):
    # returns set of stacks included in the path
    return {index[0] for index in path}

def make_aggreement_list(graph):
    '''
    graph: StackedGraph Object
    returns all stack indicies with branches or merges
    '''
    branches = []
    for idx, row in enumerate (graph.adj_matrix):
        row_counts = 0
        for stack_idx, element in enumerate(row):
            if element is not None and graph.stack_list[stack_idx].opp != 'null':
                row_counts += 1
        col_count = 0
        for stack_idx, element in enumerate(graph.adj_matrix[:, idx]):
            if element is not None and graph.stack_list[stack_idx].opp != 'null':
                col_count += 1

        if row_counts > 1 or col_count > 1:
            branches.append(idx)

    return(branches)

def get_aggreement(node_indexes, aggreement_stacks):
    '''
    returns tuple of the nodes in each of the aggreement stacks
    node_indexes: a path
    aggreement_stacks: list of stacks that must aggreee
    '''
    if not aggreement_stacks:
        return 'all'
    stack_indexes = [None for _ in aggreement_stacks]
    for node in node_indexes:
        try:
            stack_indexes[aggreement_stacks.index(node[0])] = node[1]
        except ValueError:
            continue # non-aggreement node in path

    return tuple(stack_indexes)

def ap_works(group_ap, new_ap):
    matches = True
    for idx, node in enumerate(group_ap):
        if new_ap[idx] == node or  new_ap[idx] is None  or node is None:
            continue
        else:
            matches = False

    return matches

def add_group(groups, group, stack_aggreement, cur_path, stack_coverage):
    new = False
    for idx, val in enumerate(group['ap']):
        if (val is None) != (stack_aggreement[idx] is None) and (val is None or stack_aggreement[idx]is None):
            new = True

    if new: # there are None's so keep original
        new_group = copy.deepcopy(group)
        new_group['ap'] = [(a if a is not None else b) for a, b in zip(new_group['ap'], stack_aggreement)]
        new_group['paths'] += cur_path
        new_group['coverage_groups'].append(stack_coverage)
        new_group['total_coverage'].update(stack_coverage)
        groups.append(new_group)

    else: # perfect match so mutate group in place
        group['ap'] = [(a if a is not None else b) for a, b in zip(group['ap'], stack_aggreement)]
        group['paths'] += cur_path
        group['coverage_groups'].append(stack_coverage)
        group['total_coverage'].update(stack_coverage)

def rolling_dijkstra(graph, start=(0,0)):
    '''
    Dijkstra untill there is full coverage on a combination of aggreement stacks
    graph to optimize
    '''
    aggreement_stacks = make_aggreement_list(graph)
    # all_nodes = {i.oppid for i in graph.stack_list if i.opp != 'null'}
    all_nodes = {i for i, v in enumerate(graph.stack_list) if v.opp != 'null'}

    que = []
    for stack_id in graph.load_nodes:
        que.append( (0, ( (graph.id_to_idx[stack_id],0), )) )


    groups = [] # {ap:(aggreement),paths;(), (coverage_groups:{coverage}, {groups}), total_coverage:{total coverage}}

    while que:
        cur_dist, cur_path = heapq.heappop(que) # minimum
        cur_node = cur_path[-1] # last item in path
        neighbor_stacks = graph.get_stack_neighbors(cur_node[0])

        if neighbor_stacks == []: # ending node
            stack_aggreement = get_aggreement(cur_path, aggreement_stacks)
            stack_coverage = extract_stacks(cur_path)

            added = False
            for group in groups:
                if ap_works(group['ap'], stack_aggreement) and stack_coverage not in group['coverage_groups'] and group['total_coverage'] - stack_coverage != {}: # same coverage, new path
                    add_group(groups, group, stack_aggreement, cur_path, stack_coverage)
                    added = True

            if not added:
                groups.append({ 'ap':(stack_aggreement), 'paths':tuple(cur_path), 'coverage_groups':[stack_coverage], 'total_coverage':stack_coverage })

            for group in groups:
                if group['total_coverage'] == all_nodes: # group reached full coverage:
                    return set(group['paths'])
                    # return merge_paths(group['paths'])


        for neighbor in neighbor_stacks:
            stack_connection = graph.adj_matrix[cur_node[0]][neighbor]
            for node, node_cost in enumerate(graph.stack_list[neighbor].cost_stack):
                edge_weight = stack_connection[cur_node[1]][node]
                new_distance = cur_dist + node_cost + edge_weight
                heapq.heappush(que, (new_distance, cur_path + ((neighbor, node),) ))

def graph_partition(graph):
    ''' Finds the Articulation Vertices and partitions the large graph into subgraphs
    StackedGraph objects. Inclusive on both ends of range.
    graph: StackedGraph
    '''
    groups = list(graph.get_node_groups(ASAP = False))
    assert test.group_validate(graph, groups)

    for group in groups:

        start_stack = sg.StackedNode(0, [], [[]], [[]], opp='start', func_stack=['start'], cost_stack=[0])
        first_stack = copy.deepcopy(graph.stack_list[graph.id_to_idx[group[0]]])
        first_stack.parents = [0]

        subgraph_stack_list = [start_stack, first_stack]
        for stack_id in group[1:]:
            stack = graph.stack_list[graph.id_to_idx[stack_id]]
            new_node = copy.deepcopy(stack)
            new_node.parents = set(new_node.parents) - graph.load_nodes
            subgraph_stack_list.append(new_node)


        sub_graph = sg.StackedGraph(stack_list=subgraph_stack_list)
        yield(sub_graph)

def select_nodes(graph, subgraphs):
    '''apply roling_dijkstra to each subgraph. Then apply those selections to the nodes
    of the original graph.
    graph: StackedGraph
    subgraph: StackedGraph
    such that subgraph is a partition of graph
    '''
    for idx, subgraph in enumerate(subgraphs):
        nodes = rolling_dijkstra(subgraph)
        for node in nodes:
            stack_oppid = subgraph.stack_list[node[0]].oppid

            subgraph_stack = subgraph.stack_list[subgraph.id_to_idx[stack_oppid]]
            subgraph_stack.func_selection = node[1]

            original_stack = graph.stack_list[graph.id_to_idx[stack_oppid]]
            original_stack.func_selection = node[1]

#endregion


# region ###### scheduling_dijkstra for embeded branched stacked graphs ######
def scheduling_dijkstra(graph, even=False, available_hardware = {'CPU': {'CPU1': 0, 'CPU2': 0, 'CPU3': 0}, 'PHU': {'PHU1': 0} }):
    visited = {0}
    indegree = {idx: len(stack.parents) for idx, stack in enumerate(graph.stack_list)}
    que = []
    for stack_id in graph.load_nodes:
        que.append((graph.id_to_idx[stack_id],))

    while que:
        cur_path = que.pop(0)
        cur_node = cur_path[-1]

        neighbor_stacks = graph.get_stack_neighbors(cur_node)

        for neighbor in neighbor_stacks:
            indegree[neighbor] -= 1
            if neighbor not in visited and indegree[neighbor] == 0 :
                neighbor_node = graph.stack_list[neighbor]
                hardware_type = oc.hardware_algs[neighbor_node.func_stack[neighbor_node.func_selection]][1]

                parents = [graph.stack_list[graph.id_to_idx[parent]] for parent in neighbor_node.parents]
                parent_time = max(parent.start_time + parent.cost_stack[parent.func_selection] for parent in parents)

                #select hardware
                # less_than = [available for available in available_hardware[hardware_type] if available_hardware[hardware_type][available] < parent_time*1.1] # TODO Accounts for edge weight. Fix to be real edge weight
                less_than = [available for available in available_hardware[hardware_type] if available_hardware[hardware_type][available] <= parent_time*1] # TODO Accounts for edge weight. Fix to be real edge weight
                if not less_than:
                    selected_hardware = min(available_hardware[hardware_type], key=lambda k: available_hardware[hardware_type][k])
                else:
                    if even:
                        selected_hardware = min(less_than, key=lambda k: available_hardware[hardware_type][k])
                    else:
                        selected_hardware = less_than[0]
                    available_hardware[hardware_type][selected_hardware] = parent_time


                # selected_hardware = min(available_hardware[hardware_type], key=lambda k: available_hardware[hardware_type][k])
                neighbor_node.hardware_selection = selected_hardware
                assert neighbor_node.start_time == 0
                neighbor_node.start_time = available_hardware[hardware_type][selected_hardware]

                # add time
                stack_connection = graph.adj_matrix[cur_node][neighbor]

                node_cost = neighbor_node.cost_stack[neighbor_node.func_selection]
                edge_weight = stack_connection[graph.stack_list[cur_node].func_selection][neighbor_node.func_selection]
                # available_hardware[hardware_type][selected_hardware] += node_cost + edge_weight
                available_hardware[hardware_type][selected_hardware] += node_cost
                visited.add(neighbor)
                que.append(cur_path + (neighbor,))
#endregion
