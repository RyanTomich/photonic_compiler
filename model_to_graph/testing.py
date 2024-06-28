import json
import numpy as np

import dijkstra as dijk
import stacked_graph as sg

def create_liner_stack():
    node_list = []
    a = sg.StackedNode(0, [], [[]], [[]], opp='test_opp', func_stack=['start'], cost_stack=[0])
    node_list.append(a)

    last_out = np.random.randint(low=1, high=5, size=2).tolist()

    for i in range(2):
        out_size = np.random.randint(low=1, high=5, size=2).tolist()
        cost = np.random.randint(low=1, high=5, size=3).tolist()
        node_list.append(sg.StackedNode(i+1, [i], [last_out], [out_size], opp='test_opp', func_stack=['alg1', 'alg2', 'alg3'], cost_stack=cost))
        last_out = out_size

    stacked_graph = sg.StackedGraph(stack_list = node_list)

    for node in stacked_graph.stack_list:
        print(node)

    return(stacked_graph)
    '''
    +---+
    | 0 |
    +---+
       |
       v
    +---+
    | 1 |
    +---+
       |
       v
    +---+
    | 2 |
    +---+
    '''

def create_branching_stack(print_nodes = True):
    node_list = []
    a = sg.StackedNode(0, [], [[]], [[]], opp='test_opp', func_stack=['start'], cost_stack=[0])
    node_list.append(a)

    last_out = np.random.randint(low=1, high=5, size=2).tolist()

    for i in range(4):
        out_size = np.random.randint(low=1, high=5, size=2).tolist()
        cost = np.random.randint(low=1, high=5, size=3).tolist()
        node_list.append(sg.StackedNode(i+1, [i], [last_out], [out_size], opp='test_opp', func_stack=['alg1', 'alg2', 'alg3'], cost_stack=cost))
        last_out = out_size

    out_size = np.random.randint(low=1, high=5, size=2).tolist()
    node_list[-1].oppid = 5
    node_list.append( sg.StackedNode(4, [1], node_list[1].output_shapes, [out_size], opp='test_opp', func_stack=['alg1', 'alg2', 'alg3'], cost_stack=np.random.randint(low=1, high=5, size=3).tolist()))

    node_list[4].input_shapes.append(out_size)
    node_list[4].parents.append(4)

    node_list.sort(key=lambda x: x.oppid)

    stacked_graph = sg.StackedGraph(stack_list = node_list)

    if print_nodes:
        for node in stacked_graph.stack_list:
            print(node)

    return stacked_graph
    '''
    +---+
    | 0 |
    +---+
        |
        v
    +---+
    | 1 |
    +---+
        |\
        | \
        v  v
    +---+ +---+
    | 2 | | 4 |
    +---+ +---+
        |    |
        v    |
    +---+    |
    | 3 |    |
    +---+    |
       \    /
        \  /
          v
        +---+
        | 5 |
        +---+
    '''


# stacked_graph = create_liner_stack()
# dist, previous = dijk.stacked_dijkstra(stacked_graph, (0,0))
# print(dist)
# path = dijk.stacked_get_path(previous, (len(dist)-1, np.argmin(dist[-1])) )
# print(path)


# stacked_graph = create_branching_stack(print_nodes = False)

# print(stacked_graph.get_articulation_points())
# print(dijk.branching_stacked_dijkstra(stacked_graph))

def group_validate(graph, groups):
    '''ensures every node parents are included in the group.
    exception to load and store nodes, which can have odd dependancies
    '''
    for i,lst in enumerate(groups):
        # print(f'{lst}')
        load_instructions = {stack.oppid for stack in graph.stack_list if stack.opp == 'null'}
        included = set(lst)
        for stack in lst[1:]:
            for parent in graph.stack_list[stack].parents:
                if parent in load_instructions:
                    continue
                assert parent in included
        # print(f'group {i} passed!')
    return True
