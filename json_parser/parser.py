import json
import math


with open('/home/rjtomich/photonic_compiler/Pytorch-LeNet/simple_LeNet_graph.json') as json_file:
    raw_json = json.load(json_file) # returns json file as dict

parsed_txt = open("simple_LeNet_parsed.txt", "w") # creates the write file in write mode append ('a') mode also exists

def contains(node, val):
    """recursively searches for val in each node

    Args:
        node (dict): nexted dictionary
        val (str): search word

    Returns:
        bool: True = found  word
    """
    # where node is a nesded dictionary
    if type(node) == dict:
        for key in node:
            if contains(node[key], val):
                return True
    else:
        if val in node:
            return True


def get_shape_index(node):
    return[input[0] for input in node["inputs"]]


def batch_vector(vector_size, batch_size):
    """Generator object that groups vectors into batches

    Args:
        vector_size (int): size of iriginal vector
        batch_size (int): size of each batch

    Yields:
        str: "[start_index: end index]"
    """
    temp = vector_size
    num_registers = math.ceil(vector_size / batch_size)
    start = 0
    end = batch_size
    for i in range(num_registers):
        if temp < batch_size:
            end = start + temp
        yield f"[{start}:{end}]"
        start += batch_size
        end += batch_size
        temp -= batch_size


'''
N = null
E = nonliner, electronic
P = liner, photonic

Asumes dense Multilayer perceptron. No convolutions
Should be expanede to be more robust to other formats
'''

tab = "    " # 4 space tab
max_array_len = 100

for order, node in enumerate(raw_json["nodes"]):
    # Null instructions
    if contains(node, 'null'):
        parsed_txt.write(f"N: [null] {raw_json['nodes'][order]}\n")

    # Dense instructions
    elif contains(node, 'dense'):
        parsed_txt.write(f"   [relu/MAC]{raw_json['nodes'][order]}\n")

        vector_index, matrix_index = get_shape_index(node)
        vector = raw_json['attrs']['shape'][1][vector_index]
        matrix = raw_json['attrs']['shape'][1][matrix_index]

        # vector registration
        parsed_txt.write(f"   {tab}[read] {vector_index}, {matrix_index}\n") # indicies
        num_registers = math.ceil(vector[1] / max_array_len)
        batch_gen = batch_vector(vector[1], max_array_len)

        for register in range(num_registers):
            parsed_txt.write(f"E:  {tab*2}load vector: a{register}, {1}{next(batch_gen)}\n")

        # Dot products
        parsed_txt.write(f"   {tab}[MAC] {vector} x {matrix}\n")
        for matrix_row in range(matrix[0]):
            parsed_txt.write(f"P: {tab*2}{vector} . {matrix}[{matrix_row}]\n")

            batch_gen = batch_vector(vector[1], max_array_len)
            to_sum = []
            for register in range(num_registers):
                to_sum.append(register + num_registers)
                parsed_txt.write(f"E:  {tab*3}load vector: a{register + num_registers}, {matrix_index}{next(batch_gen)}\n")
                parsed_txt.write(f"P:  {tab*3}MAC: a{register + num_registers}, a{register + num_registers}, a{register}\n")

            accumulate = to_sum.pop(0)
            for i in to_sum:
                parsed_txt.write(f"E:  {tab*3}add: a{accumulate}, a{i}\n")

            parsed_txt.write(f"E:  {tab*3}save:{vector}[{matrix_row}], a{accumulate}\n")

        parsed_txt.write(f"E: {tab}[relu] {vector}\n")

    # Catch all
    else:
        parsed_txt.write(f"E: [other] {raw_json['nodes'][order]}\n")
        input_index = get_shape_index(node)
        if input_index:
            parsed_txt.write(f"E:  {tab}[read] {input_index}\n")


# # Nicer visualization
# if "nodes" in raw_json:
#     for num, node in enumerate(raw_json['nodes']):
#         parsed_txt.write(f"nodes[{num}]:\n")
#         for key in node:
#             if key == "attrs":
#                 parsed_txt.write(f"         {key}: \n")
#                 for attribute in node[key]:
#                     parsed_txt.write(f"           {attribute}: {node[key][attribute]}\n")
#                 continue
#             parsed_txt.write(f"         {key}: {node[key]}\n")


#         # print(f"nodes[{num}]: {node}")
#         # print("")

# if "attrs" in raw_jsopn:
#     attrs = raw_json["attrs"]
#     if "shape" in attrs:
#         parsed_txt.write(f"Shape attributes: {attrs['shape'][1]}")
