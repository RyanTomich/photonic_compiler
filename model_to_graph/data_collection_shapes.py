import json

import hardware as hw
import dijkstra as dijk
import stacked_graph as sg
import testing as test
import data_collection as dc
import code_generation as cg

relay_path = "/home/rjtomich/photonic_compiler/validation/modles/gpt2_graph.json"

with open(relay_path, encoding="utf-8") as json_file:
    raw_json = json.load(json_file)  # returns json file as dict
    # print("... Json loaded ...")

WEIGHT_VARIABLE = 'time'

graph = sg.StackGraph(raw_json=raw_json, weight_variable=WEIGHT_VARIABLE)

shapes = {}
types = {}
for stack in graph.stack_list:
    if stack.tvm_func not in shapes:
        types[stack.tvm_func] = stack.opp
        shapes[stack.tvm_func] = []
        shapes[stack.tvm_func].append(stack.input_shapes)
        shapes[stack.tvm_func].append(stack.output_shapes)

for name, size in shapes.items():
    print(f'{name if name else 'none':<60} {types[name]:<10} {str(size):<30}')
