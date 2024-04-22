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


#region Metrics

class MetricsCounter:
    def __init__(self):
        self.MAC_instructions = 0
        self.loadvec_instructions = 0
        self.add_instructions = 0
        self.save_instructions = 0
        self.reg_used = 0

    def increment(self, instruction_type):
        if instruction_type == 'add':
            self.add_instructions += 1
        elif instruction_type == 'MAC':
            self.MAC_instructions += 1
        elif instruction_type == 'load_vector':
            self.loadvec_instructions += 1
        elif instruction_type == 'save':
            self.save_instructions += 1
    def num_reg_used(self, max_register):
        self.reg_used = max(self.reg_used, max_register)

    def __repr__(self):
        return (f"MAC Instructions: {self.MAC_instructions}\n"
                f"Load vector Instructions: {self.loadvec_instructions}\n"
                f"Add Instructions: {self.add_instructions}\n"
                f"Save Instructions: {self.save_instructions}\n"
                f"Registers Used: {self.reg_used + 1}") # account for 0 register.

metrics_counter = MetricsCounter()

def metrics_counter_dec(func):
    def wrapper(instruction_type, *args, **kwargs):
        metrics_counter.increment(instruction_type)

        # Extracting the max register used
        if instruction_type in ['add', 'MAC']:
            a1, a2, a3 = args
            max_register = max(a1, a2, a3)
            metrics_counter.num_reg_used(max_register)
        elif instruction_type == 'load_vector':
            a1 = args[0]
            metrics_counter.num_reg_used(a1)
        elif instruction_type == 'save':
            a1 = args[2]
            metrics_counter.num_reg_used(a1)

        result =  func(instruction_type, *args, **kwargs)
        return result
    return wrapper

#endregion

#region File access
current_directory = os.path.dirname(os.path.abspath(__file__))

read_json_path = os.path.join(current_directory, '..', 'Pytorch-LeNet', 'simple_LeNet_graph.json')
with open(read_json_path)  as json_file:
    raw_json = json.load(json_file) # returns json file as dict

output_file_path = os.path.join(current_directory, 'simple_LeNet_parsed.txt')
parsed_txt = open(output_file_path, "w") # creates the write file in write mode append ('a') mode also exists
#endregion

#region Node helper Functions

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


@metrics_counter_dec
def write_instruction(instruction_type, *args):
    if instruction_type == 'add':
        a1, a2, a3 = args
        parsed_txt.write(f"E: add: a{a1}, a{a2}, a{a3}\n")

    elif instruction_type == "MAC":
        a1, a2, a3 = args
        parsed_txt.write(f"P: MAC: a{a1}, a{a2}, a{a3}\n")

    elif instruction_type == 'load_vector':
        a1, matrix, matrix_index, batch = args
        parsed_txt.write(f"E: load vector: a{a1}, {matrix}[{matrix_index}]{batch}\n")

    elif instruction_type == 'save':
        vector, matrix_row, a1 = args
        parsed_txt.write(f"E: save:{vector}[{matrix_row}], a{a1}\n")


def sequential_cross_product(node):
    """Prioritized finishing each row of output entirely before moving to next. Breaks crossproduct into instructions.
    Args:
        node (dict): one node of thejson graph representing a layer and relu
    """

    # vector registration
    parsed_txt.write(f"   [read] {vector_index}, {matrix_index}\n") # indicies
    num_needed_registers = math.ceil(vector[1] / max_array_len) # number of registers needed to hold vector
    batch_gen = batch_vector(vector[1], max_array_len) # generator object seperating each vector slice

    for register in range(num_needed_registers):
        write_instruction('load_vector', register, vector_index, 1, next(batch_gen))

    # Dot products
    parsed_txt.write(f"   [MAC] {vector} x {matrix}\n")
    accumulate_register = register + 1
    parsed_txt.write(f"   Accumulate register: a{accumulate_register}\n")

    for matrix_row in range(matrix[0]):
        parsed_txt.write(f"   {vector} . {matrix}[{matrix_row}]\n")

        batch_gen = batch_vector(vector[1], max_array_len)
        working_register = accumulate_register + 1
        for register in range(num_needed_registers):
            write_instruction('load_vector', working_register, matrix_index, matrix_row, next(batch_gen))
            write_instruction('MAC', working_register, working_register, register)
            write_instruction('add', accumulate_register, accumulate_register, working_register)

        write_instruction('save', vector, matrix_row, accumulate_register)

    parsed_txt.write(f"E: [relu] {vector}\n")


def concurrent_cross_product(node):
    parsed_txt.write(f"   [read] {vector_index}, {matrix_index}\n") # indicies
    num_needed_registers = math.ceil(vector[1] / max_array_len) # number of registers needed to hold vector
    batch_gen = batch_vector(vector[1], max_array_len) # generator object seperating each vector slice
    parsed_txt.write(f"   [MAC] {vector} x {matrix}\n")



    for _ in range(num_needed_registers):
        batch_index = next(batch_gen)
        write_instruction('load_vector', 0, vector_index,1, batch_index)
        for matrix_row in range(matrix[0]):
            write_instruction('load_vector', 1, matrix_index, matrix_row, batch_index)
            write_instruction('MAC', 1, 0, 1)
            write_instruction('save', vector, matrix_row, 1)


#endregion

# Loop over Nodes

max_array_len = 100
concurrent = False


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

        if concurrent == False:
            sequential_cross_product(node)
        else:
            concurrent_cross_product(node)

    # Catch all
    else:
        parsed_txt.write(f"E: [other] {raw_json['nodes'][order]}\n")
        input_index = get_shape_index(node)
        if input_index:
            parsed_txt.write(f"   [read] {input_index}\n")



print(metrics_counter)
