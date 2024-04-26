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
import matplotlib.pyplot as plt


#region Metrics

class MetricsCounter:
    def __init__(self):
        self.MAC_instructions = 0
        self.add_instructions = 0
        self.save_instructions = 0

    def increment(self, instruction_type, amount = 1):
        if instruction_type == 'add':
            self.add_instructions += 1
        elif instruction_type == 'MAC':
            self.MAC_instructions += 1
        elif instruction_type == 'save':
            self.save_instructions += 1

    def plot_add_data(self):
        MAC_instructions_plot.append(self.MAC_instructions)
        add_instructions_plot.append(self.add_instructions)
        save_instructions_plot.append(self.save_instructions)

    def __repr__(self):
        return (f"MAC Instructions: {self.MAC_instructions}\n"
                f"Add Instructions: {self.add_instructions}\n"
                f"Save Instructions: {self.save_instructions}\n")


def metrics_counter_dec(func):
    def wrapper(instruction_type, *args, **kwargs):
        metrics_counter.increment(instruction_type)
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


def batch_vector(vector_size, num_batches):
    """Generator object that groups vectors into batches
    Args:
        vector_size (int): size of original vector
        batch_size (int): num hardware
    Yields:
        str: "[start_index: end index]"
    """
    temp = vector_size
    batch_size = math.ceil(vector_size/ num_batches)
    start = 0
    end = batch_size
    for i in range(num_batches):
        if temp < batch_size:
            end = start + temp
        yield [start, end]
        start += batch_size
        end += batch_size
        temp -= batch_size


def looper(num_photonic_hardware):
    """contunoously loops over a numer n
        1,2,3,1,2,3,1,2,3 ...
    Args: num_photonic_hardware (int): loop length
    Yields: int: num between 1 to num_photonic_hardware
    """
    while True:
        for i in range(1, num_photonic_hardware+1, 1):
            yield i


@metrics_counter_dec
def write_instruction(instruction_type, *args):
    if instruction_type == 'add':
        a1, a2, a3 = args
        parsed_txt.write(f"E: add: a{a1}, a{a2}, a{a3}\n")

    elif instruction_type == "MAC":
        computerID, write, v1, v2 = args
        parsed_txt.write(f"P{computerID}: MAC: {write}, {v1}, {v2}\n")

    elif instruction_type == 'load_vector':
        a1, matrix, matrix_index, batch = args
        parsed_txt.write(f"E: load vector: a{a1}, {matrix}[{matrix_index}]{batch}\n")

    elif instruction_type == 'save':
        write, read = args
        parsed_txt.write(f"E: save:{write}, {read}\n")


def opt_strat(node, optimization):

    def task_parallel(num_photon_hardware, node):
        # one hardware per matrix row
        for matrix_row in range(matrix[0]):
            write = f'[1:{matrix[0]}][{matrix_row}]'
            v1 = f'{vector}'
            v2 = f'{matrix}[{matrix_row}]'
            write_instruction('MAC',next(P_computer_num_gen), write, v1, v2)

    def data_parrellel(num_photon_hardware,node):
        for matrix_row in range(matrix[0]):
            # generator object seperating each vector slice
            batch_gen = batch_vector(matrix[1], num_photon_hardware)
            for batch in batch_gen:
                batch = f"[{batch[0]}:{batch[1]}]"
                v1 = f"{vector}{batch}"
                v2 = f"{matrix}[{matrix_row}]{batch}"
                write_instruction('MAC',next(P_computer_num_gen), 'a1', v1, v2)
                write_instruction("add",0, 0, 1)
            write = f'[1:{matrix[0]}][{matrix_row}]'
            write_instruction("save",write, 'a0')

    vector_index, matrix_index = get_shape_index(node)
    vector = raw_json['attrs']['shape'][1][vector_index]
    matrix = raw_json['attrs']['shape'][1][matrix_index]

    parsed_txt.write(f"   [read] {vector_index}, {matrix_index}\n") # indicies
    parsed_txt.write(f"   [MAC] {vector} x {matrix}\n")
    P_computer_num_gen = looper(num_photon_hardware)

    if optimization =='task_parrellel':
        task_parallel(num_photon_hardware, node)
    elif optimization == 'data_parrellel':
        data_parrellel(num_photon_hardware, node)

    parsed_txt.write(f"E: [relu] [1:{matrix[0]}]\n")

#endregion


# Loop over Nodes
def main_loop(num_photon_hardware, optimization = ""):
    for order, node in enumerate(raw_json["nodes"]):
        # Null instructions - N:
        if contains(node, 'null'):
            parsed_txt.write(f"N: [null] {raw_json['nodes'][order]}\n")

        # Dense instructions
        elif contains(node, 'dense'):
            parsed_txt.write(f"   [relu/MAC]{raw_json['nodes'][order]}\n")
            opt_strat(node, optimization)
        # Catch all
        else:
            parsed_txt.write(f"E: [other] {raw_json['nodes'][order]}\n")
            input_index = get_shape_index(node)
            if input_index:
                parsed_txt.write(f"   [read] {input_index}\n")


    metrics_counter.plot_add_data()


MAC_instructions_plot = []
add_instructions_plot = []
save_instructions_plot = []
cycles_plot = []
num_photonic_hardware_plot = []


metrics_counter = MetricsCounter()
num_photon_hardware = 300
main_loop(num_photon_hardware, optimization = "data_parrellel")
print(metrics_counter)

# region plotting
# for num_photon_hardware in range(10, 1000, 10): # 1000 photonic hardware
#     metrics_counter = MetricsCounter()
#     num_photonic_hardware_plot.append(num_photon_hardware)
#     main_loop(num_photon_hardware, optimization = "task_parrellel")


# plt.plot(num_photonic_hardware_plot,MAC_instructions_plot, label = "MAC's")
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.legend()
# plt.title('Simple Plot')

# # plt.show()
# plt.savefig('plot.png')

#endregion
