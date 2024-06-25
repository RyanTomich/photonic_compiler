import numpy as np
import heapq
import stacked_graph as sg

#region ###### General Dijkstra for normal graph ######
dependancies = [
    (0, 1, 1), (0, 2, 2), (0,3,2),

    (1,4, 2), (1,5,1), (1,6,2),
    (2,4,2), (2,5,2), (2,6,2),
    (3,4,2), (3,5,2), (3,6,2),

    (4,7,2), (5,7,1), (6,7,2),
]

values = [1, 10, 100, 100, 100, 10, 100, 10]
# make adj matrix
adj_matrix = np.zeros((len(values), len(values)))
for dep in dependancies:
    adj_matrix[dep[0]][dep[1]] = dep[2]

# print(adj_matrix)

def matrix_dijkstra(adj_matrix, node_weights, start = 0):
    n = len(node_weights)
    dist = [np.inf] * n
    dist[start] = node_weights[start]
    visited = [False]*n
    que = [(node_weights[start], start)] # (distance, node)
    previous = [-1] * n

    while que:
        cur_dist, u = heapq.heappop(que)

        if visited[u]:
            continue

        visited[u] = True

        for v in range(n):
            if adj_matrix[u][v] != 0 and not visited[v]:
                new_dist = cur_dist + adj_matrix[u][v] + node_weights[v]

                if new_dist < dist[v]:
                    dist[v] = new_dist
                    heapq.heappush(que, (new_dist, v))
                    previous[v] = u

    return dist, previous



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
# make pretend tree
# node_list = []
# a = sg.StackedNode(0, [], [[]], [[]], opp='matmul', func_stack=['start'], cost_stack=[0])
# node_list.append(a)

# last_out = np.random.randint(low=1, high=5, size=2).tolist()

# for i in range(2):
#     out_size = np.random.randint(low=1, high=5, size=2).tolist()
#     cost = np.random.randint(low=1, high=5, size=3).tolist()
#     node_list.append(sg.StackedNode(i+1, [i], [last_out], [out_size], opp='matmul', func_stack=['alg1', 'alg2', 'alg3'], cost_stack=cost))
#     last_out = out_size

# stacked_graph = sg.StackedGraph(stack_list = node_list)

# for node in stacked_graph.stack_list:
#     print(node)


# Path finding
def stacked_dijkstra(graph, start):
    node_matrix = graph.make_node_matrix()

    dist = np.copy(node_matrix)
    dist[dist == 1] = np.inf
    dist[start] = graph.stack_list[start[0]].cost_stack[start[1]]

    visited = np.copy(node_matrix)
    visited[visited == 1] = 0

    previous = np.full(node_matrix.shape, np.nan, dtype=object)
    non_zero = np.argwhere(node_matrix == 1)
    for idx in non_zero:
        previous[tuple(idx)] = (-1, -1)

    que = [(dist[start], start)] # (distance, node)

    while que:
        cur_dist, cur_position = heapq.heappop(que)
        if visited[cur_position]:
            continue

        visited[cur_position] = True

        for board_position in np.ndindex(node_matrix.shape):
            # if not in visited and their stacks the nodes stacks are connected
            stack_connection = graph.adj_matrix[cur_position[0]][board_position[0]]
            if visited[board_position] == 0 and stack_connection is not None:
                next_node_weight = graph.stack_list[board_position[0]].cost_stack[board_position[1]]
                edge_weight = stack_connection[cur_position[1]][board_position[1]]
                new_dist = cur_dist + edge_weight + next_node_weight

                if new_dist < dist[board_position]:
                    dist[board_position] = new_dist
                    heapq.heappush(que, (new_dist, board_position))
                    previous[board_position] = cur_position

    return dist, previous

def stacked_get_path(previous, target):
    print(target)
    path = []
    while target != (-1, -1):
        path.insert(0, target)
        target = previous[target]
    return path


# dist, previous = stacked_dijkstra(stacked_graph, (0,0))
# print(dist)

# path = stacked_get_path(previous, (len(dist)-1, np.argmin(dist[-1])) )
# print(path)


#endregion

# region ###### Dijkstra for branched stacked graphs ######

node_list = []
a = sg.StackedNode(0, [], [[]], [[]], opp='matmul', func_stack=['start'], cost_stack=[0])
node_list.append(a)

last_out = np.random.randint(low=1, high=5, size=2).tolist()

for i in range(4):
    out_size = np.random.randint(low=1, high=5, size=2).tolist()
    cost = np.random.randint(low=1, high=5, size=3).tolist()
    node_list.append(sg.StackedNode(i+1, [i], [last_out], [out_size], opp='matmul', func_stack=['alg1', 'alg2', 'alg3'], cost_stack=cost))
    last_out = out_size

out_size = np.random.randint(low=1, high=5, size=2).tolist()
node_list[-1].oppid = 5
node_list.append( sg.StackedNode(4, [1], node_list[1].output_shapes, [out_size], opp='matmul', func_stack=['alg1', 'alg2', 'alg3'], cost_stack=np.random.randint(low=1, high=5, size=3).tolist()))

node_list[4].input_shapes.append(out_size)
node_list[4].parents.append(4)

node_list.sort(key=lambda x: x.oppid)

stacked_graph = sg.StackedGraph(stack_list = node_list)

# for node in stacked_graph.stack_list:
#     print(node)

def get_combinations(graph, aggreement_stacks):
    '''
    returns all the combinations of nodes within the agreed stacks
    graph: StackedGraph
    aggreement_stacks: list of stacks that must aggree
    '''
    ans = []
    lengths = [len(graph.stack_list[stack].func_stack) for stack in aggreement_stacks]
    for i in range(lengths[0]):
        for j in range(lengths[1]):
            ans.append((i,j))
    return ans

def get_aggreement(node_indexes, aggreement_stacks):
    '''
    returns tuple of the nodes in each of the aggreement stacks
    node_indexes: a path
    aggreement_stacks: list of stacks that must aggreee
    '''
    stack_indexes = ()
    for node in node_indexes:
        if node[0] in aggreement_stacks:
            stack_indexes += (node[1],)
    return stack_indexes

def extract_stacks(path):
    # returns set of stacks included in the path
    return {index[0] for index in path}

def merge_paths(paths):
    '''
    given a sorted by time list of paths with full coverage, returns all nodes in
    the optimal path.
    [(distance, [path]), (distance, [path]), (distance, [path])]
    '''
    row_coverage = set()
    nodes = []
    for path in paths:
        for node in path[1]:
            if node[0] not in row_coverage:
                nodes.append(node)
                row_coverage.add(node[0])
    return sorted(list(nodes), key=lambda x: x[0])

def make_aggreement_list(graph):
    '''
    graph: StackedGraph Object
    returns all stack indicies with branches or merges
    '''
    branches = []
    for idx, row in enumerate (graph.adj_matrix):
        row_counts = 0
        for element in row:
            if element is not None:
                row_counts += 1
        col_count = 0
        for element in graph.adj_matrix[:, idx]:
            if element is not None:
                col_count += 1

        if row_counts > 1 or col_count > 1:
            branches.append(idx)
    return(branches)



def branching_stacked_dijkstra(graph, start):
    '''
    Returns optimal path covering all stacks
    graph: StackedGraph object
    start: starting node (stack, node)
    '''
    aggreement_stacks = make_aggreement_list(graph)
    print(aggreement_stacks)
    node_matrix = graph.make_node_matrix()

    que = [(0, [start]) ] #(distance, [path])
    paths = {x : [] for x in get_combinations(graph, aggreement_stacks)}
    coverage = {x : {i for i in range(len(node_matrix))} for x in get_combinations(graph, aggreement_stacks)}

    while que:
        cur_dist, cur_path = heapq.heappop(que) # minimum
        cur_node = cur_path[-1] # last item in path
        negibors = graph.get_stack_neighbors(cur_node[0])
        if negibors == []: # ending node
            aggreement = get_aggreement(cur_path, aggreement_stacks)
            # Paths come in ordered, so only keep paths that add new stacks.
            if extract_stacks(cur_path) - coverage[aggreement] != {}: # There are unique stacks in current path
                paths[aggreement].append((cur_dist, cur_path))
                coverage[aggreement] -= extract_stacks(cur_path)
            if not coverage[aggreement]: # That combination of node stack aggreement has full coverage.
                return merge_paths(paths[aggreement])

        for negibor_stack in negibors:
            stack_connection = graph.adj_matrix[cur_node[0]][negibor_stack]
            for node, node_cost in enumerate(graph.stack_list[negibor_stack].cost_stack):
                edge_weight = stack_connection[cur_node[1]][node]
                # new_distance = cur_dist + sum(node_cost.values()) + edge_weight
                new_distance = cur_dist +node_cost + edge_weight
                heapq.heappush(que, (new_distance, cur_path + [(negibor_stack, node)]))


print(branching_stacked_dijkstra(stacked_graph, (0,0)))

# endregion
