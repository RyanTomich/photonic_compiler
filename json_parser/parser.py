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
    def __init__(self, opp):
        self.optimization = opp
        self.MAC_instructions = 0
        self.sum_instructions = 0
        self.save_instructions = 0
        self.time = 0

    def increment(self, instruction_type, amount = 1):
        if instruction_type == 'sum':
            self.sum_instructions += 1
        elif instruction_type == 'MAC':
            self.MAC_instructions += 1
        elif instruction_type == 'save':
            self.save_instructions += 1
        elif instruction_type == 'time':
            self.time += amount

    def plot_add_data(self):
        MAC_instructions_plot.append(self.MAC_instructions)
        sum_instructions_plot.append(self.sum_instructions)
        save_instructions_plot.append(self.save_instructions)
        time_plot.append(self.time)

    def __str__(self):
        return (f"optimization: {self.optimization}\n"
                f"MAC Instructions: {self.MAC_instructions}\n"
                f"Sum Instructions: {self.sum_instructions}\n"
                f"Save Instructions: {self.save_instructions}\n"
                f"time: {self.time}")


def metrics_counter_dec(func):
    def wrapper(instruction_type, *args, **kwargs):
        metrics_counter.increment(instruction_type)
        result = func(instruction_type, *args, **kwargs)
        return result
    return wrapper

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
    remainder = vector_size % num_batches
    start = 0
    end = batch_size
    for i in range(num_batches):
        if remainder == 0:
            batch_size = batch_size-1
            end = end-1
        # if temp < batch_size:
        #     end = start + temp
        yield [start, end]
        start += batch_size
        end += batch_size
        temp -= batch_size
        remainder -= 1


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
    if instruction_type == 'null':
        order = int(args[0])
        if write_to_file: parsed_txt.write(f"N: [null] {raw_json['nodes'][order]}\n")

    if instruction_type == 'other':
        order = int(args[0])
        if write_to_file: parsed_txt.write(f"E: [other] {raw_json['nodes'][order]}\n")

    elif instruction_type == 'dense':
        order = int(args[0])
        if write_to_file: parsed_txt.write(f"   [relu/MAC]{raw_json['nodes'][order]}\n")

    elif instruction_type == 'other':
        order = int(args[0])
        if write_to_file: parsed_txt.write(f"E: [other] {raw_json['nodes'][order]}\n")

    elif instruction_type == 'sum':
        a1, a2, a3 = args
        if write_to_file: parsed_txt.write(f"E: sum: a{a1}, [a{a2}: a{a3}]\n")

    elif instruction_type == "MAC":
        computerID, write, v1, v2, size = args
        if computerID == 1:
            metrics_counter.increment('time', size * 10**-10) # photonic is 10 Ghz
        if write_to_file: parsed_txt.write(f"P{computerID}: MAC: {write}, {v1}, {v2}\n")

    elif instruction_type == 'load_vector':
        a1, matrix, matrix_index, batch = args
        if write_to_file: parsed_txt.write(f"E: load vector: a{a1}, {matrix}[{matrix_index}]{batch}\n")

    elif instruction_type == 'save':
        write, read = args
        if write_to_file: parsed_txt.write(f"E: save:{write}, {read}\n")


def opt_strat(node, optimization):

    def task_parallel(num_photon_hardware, node):
        # one hardware per matrix row
        for matrix_row in range(matrix[0]):
            write = f'[1:{matrix[0]}][{matrix_row}]'
            v1 = f'{vector}'
            v2 = f'{matrix}[{matrix_row}]'
            size = vector[1]
            write_instruction('MAC',next(P_computer_num_gen), write, v1, v2, size)

    def data_parrellel(num_photon_hardware,node):
        for matrix_row in range(matrix[0]):
            # generator object seperating each vector slice
            batch_gen = batch_vector(matrix[1], num_photon_hardware)
            largest_batch = 0
            for batch in batch_gen:
                size = batch[1]-batch[0]
                largest_batch = max(largest_batch, size)
                batch = f"[{batch[0]}:{batch[1]}]"
                v1 = f"{vector}{batch}"
                v2 = f"{matrix}[{matrix_row}]{batch}"
                photonic_hardware_id = next(P_computer_num_gen)
                write_instruction('MAC',photonic_hardware_id, f'a{photonic_hardware_id-1}', v1, v2, size)
            write_instruction("sum", 0, 0, num_photon_hardware)
            write = f'[1:{matrix[0]}][{matrix_row}]'
            write_instruction("save",write, 'a0')

        metrics_counter.increment('time', math.log2(num_photon_hardware)* 10**-8)  #-8 electronic is 0.1 Ghz

    # def dynamic_parallel()
    #     pass #TODO


    optimization_algs = {'task_para': task_parallel, 'data_para': data_parrellel}

    vector_index, matrix_index = get_shape_index(node)
    vector = raw_json['attrs']['shape'][1][vector_index]
    matrix = raw_json['attrs']['shape'][1][matrix_index]

    if write_to_file:
        parsed_txt.write(f"   [read] {vector_index}, {matrix_index}\n") # indicies
        parsed_txt.write(f"   [MAC] {vector} x {matrix}\n")

    P_computer_num_gen = looper(num_photon_hardware)
    optimization_algs[optimization](num_photon_hardware, node)

    if write_to_file: parsed_txt.write(f"E: [relu] [1:{matrix[0]}]\n")

#endregion


# Loop over Nodes
def main_loop(num_photon_hardware, optimization = ""):
    for order, node in enumerate(raw_json["nodes"]):
        if contains(node, 'null'):  # Null instructions - N:
            write_instruction('null', order)

        elif contains(node, 'dense'): # Dense instructions
            write_instruction('dense', order)
            opt_strat(node, optimization)

        else: # Catch all
            write_instruction('other', order)

    metrics_counter.plot_add_data()


#region File access
current_directory = os.path.dirname(os.path.abspath(__file__))

read_json_path = os.path.join(current_directory, '..', 'Pytorch-LeNet', 'simple_LeNet_graph.json')
with open(read_json_path)  as json_file:
    raw_json = json.load(json_file) # returns json file as dict

output_file_path = os.path.join(current_directory, 'simple_LeNet_parsed.txt')
parsed_txt = open(output_file_path, "w") # creates the write file in write mode append ('a') mode also exists
#endregion


MAC_instructions_plot = []
sum_instructions_plot = []
save_instructions_plot = []
time_plot = []
num_photonic_hardware_plot = []

num_photon_hardware = 100
write_to_file = True

# opt = 'task_para'
# metrics_counter = MetricsCounter(opt)
# main_loop(num_photon_hardware, optimization = opt)
# print(metrics_counter)
# print('\n')

# opt = 'data_para'
# metrics_counter = MetricsCounter(opt)
# main_loop(num_photon_hardware, optimization = opt)
# print(metrics_counter)


# region plotting
optimizations = ['task_para', 'data_para']
for opt in optimizations:
    MAC_instructions_plot = []
    add_instructions_plot = []
    save_instructions_plot = []
    time_plot = []
    num_photonic_hardware_plot = []
    write_to_file = False

    for num_photon_hardware in range(10, 1000, 10): # 1000 photonic hardware
        metrics_counter = MetricsCounter(opt)
        num_photonic_hardware_plot.append(num_photon_hardware)
        main_loop(num_photon_hardware, optimization = opt)

    # plt.plot(num_photonic_hardware_plot,MAC_instructions_plot, label = f"MAC's: {opt}")
    # plt.plot(num_photonic_hardware_plot,add_instructions_plot, label = f"ADD's: {opt}")
    # plt.plot(num_photonic_hardware_plot,save_instructions_plot, label = f"SAVE's: {opt}")
    plt.plot(num_photonic_hardware_plot,time_plot, label = f"time: {opt}")

plt.xlabel('# photonic Hardware')
plt.ylabel('')
plt.legend()
plt.title(f'both optimization (overlap time)')

plt.savefig('plot.png')
#endregion
