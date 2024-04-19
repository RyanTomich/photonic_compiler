'''
Author: Ryan Tomich
Project: Photonic-Compiler: Dense Liner Parser

N = null
E = nonliner, electronic
P = liner, photonic

Asumes dense Multilayer perceptron. No convolutions
'''

import json
import math
import os



# Metrics
MAC_instructions = 0
loadvec_instructions = 0
add_instructions = 0
save_instructions = 0

# File access
current_directory = os.path.dirname(os.path.abspath(__file__))

read_json_path = os.path.join(current_directory, '..', 'Pytorch-LeNet', 'simple_LeNet_graph.json')
with open(read_json_path)  as json_file:
    raw_json = json.load(json_file) # returns json file as dict

output_file_path = os.path.join(current_directory, 'simple_LeNet_parsed.txt')
parsed_txt = open(output_file_path, "w") # creates the write file in write mode append ('a') mode also exists


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
    num_needed_registers = math.ceil(vector_size / batch_size)
    start = 0
    end = batch_size
    for i in range(num_needed_registers):
        if temp < batch_size:
            end = start + temp
        yield f"[{start}:{end}]"
        start += batch_size
        end += batch_size
        temp -= batch_size

tab = "    " # 4 space tab
max_array_len = 100


for order, node in enumerate(raw_json["nodes"]):
    # Null instructions - N:
    if contains(node, 'null'):
        parsed_txt.write(f"N: [null] {raw_json['nodes'][order]}\n")

    # Dense instructions
    elif contains(node, 'dense'):
        parsed_txt.write(f"   [relu/MAC]{raw_json['nodes'][order]}\n")

        # get vector index and shapes
        vector_index, matrix_index = get_shape_index(node)
        vector = raw_json['attrs']['shape'][1][vector_index]
        matrix = raw_json['attrs']['shape'][1][matrix_index]

        # vector registration
        parsed_txt.write(f"   {tab}[read] {vector_index}, {matrix_index}\n") # indicies
        num_needed_registers = math.ceil(vector[1] / max_array_len) # number of registers needed to hold vector
        batch_gen = batch_vector(vector[1], max_array_len) # generator object seperating each vector slice

        for register in range(num_needed_registers):
            parsed_txt.write(f"E:  {tab*2}load vector: a{register}, {1}{next(batch_gen)}\n")

        # Dot products
        parsed_txt.write(f"   {tab}[MAC] {vector} x {matrix}\n")
        accumulate_register = register + 1
        parsed_txt.write(f"    {tab*2}Accumulate register: a{accumulate_register}\n")

        for matrix_row in range(matrix[0]):
            parsed_txt.write(f"   {tab*2}{vector} . {matrix}[{matrix_row}]\n")

            batch_gen = batch_vector(vector[1], max_array_len)
            working_register = accumulate_register + 1
            for register in range(num_needed_registers):
                parsed_txt.write(f"E:  {tab*3}load vector: a{working_register}, {matrix_index}{next(batch_gen)}\n")
                parsed_txt.write(f"P:  {tab*3}MAC: a{working_register}, a{working_register}, a{register}\n")
                parsed_txt.write(f"E:  {tab*3}add: a{accumulate_register}, a{accumulate_register}, a{working_register}\n")


            parsed_txt.write(f"E:  {tab*3}save:{vector}[{matrix_row}], a{accumulate_register}\n")

        parsed_txt.write(f"E: {tab}[relu] {vector}\n")

    # Catch all
    else:
        parsed_txt.write(f"E: [other] {raw_json['nodes'][order]}\n")
        input_index = get_shape_index(node)
        if input_index:
            parsed_txt.write(f"E:  {tab}[read] {input_index}\n")
