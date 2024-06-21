import numpy as np
import heapq
import stacked_graph as sg

# General Dijkstra for normal graph
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

print(adj_matrix)

def dijkstra(adj_matrix, node_weights, start = 0):
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


def get_path(previous, target):
    path = []
    while target != -1:
        path.insert(0, target)
        target = previous[target]
    return path

# dist, previous = dijkstra(adj_matrix, values, start = 0)
# print(dist)
# print(previous)
# print(get_path(previous, len(adj_matrix)-1))


# Special Dijkstra for liner stacked graphs
node_list = []
a = sg.StackedNode(0, [], [[]], [[]], opp='matmul', func_stack=['start'], cost_stack=[0])
node_list.append(a)

last_out = np.random.randint(low=1, high=5, size=2).tolist()

for i in range(2):
    out_size = np.random.randint(low=1, high=5, size=2).tolist()
    cost = np.random.randint(low=1, high=5, size=3).tolist()
    node_list.append(sg.StackedNode(i+1, [i], [last_out], [out_size], opp='matmul', func_stack=['alg1', 'alg2', 'alg3'], cost_stack=cost))
    last_out = out_size

stacked_graph = sg.StackedGraph(node_list = node_list)


for node in stacked_graph.node_list:
    print (node)
print(stacked_graph.adj_matrix)
