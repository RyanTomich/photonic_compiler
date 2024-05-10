'''
Author: Ryan Tomich
Project: Photonic-Compiler: Dense Liner Parser
'''

import json
import math
import os
import matplotlib.pyplot as plt



#region Metrics

class MetricsCounter:
    """Keeps track of metrics with incrament, plot, print"""
    def __init__(self, opp):
        self.optimization = opp
        self.num_write = 0
        self.num_read = 0
        self.MAC_instructions = 0
        self.sum_instructions = 0
        self.save_instructions = 0
        self.registers = 0
        self.time = 0

        self.instruction_types = {
            'read': 'num_write',
            'write': 'num_write',
            'sum': 'sum_instructions',
            'MAC': 'MAC_instructions',
            'save': 'save_instructions',
            'register': 'registers',
            'time': 'time'
        }

    def increment(self, instruction_type, function=sum, amount=1):
        """add amount to the metric
        Args:
            instruction_type (str): from the instruction_types dict
            function (function): what to do with new and old metric. Defaults to sum.
            amount (int): the amount to incrament. Defaults to 1.
        """
        if instruction_type in self.instruction_types:
            attr_name = self.instruction_types[instruction_type]
            current_value = getattr(self, attr_name)
            new_value = function((current_value, amount))
            setattr(self, attr_name, new_value)


    def plot_add_data(self):
        """appends all metrics to globally defined plots"""
        num_read_plot.append(self.num_read)
        num_write_plot.append(self.num_write)
        MAC_instructions_plot.append(self.MAC_instructions)
        sum_instructions_plot.append(self.sum_instructions)
        save_instructions_plot.append(self.save_instructions)
        registers_plot.append(self.registers)
        time_plot.append(self.time)

    def __str__(self):
        return (f"optimization: {self.optimization}\n"
                f"num_read: {self.num_read}\n"
                f"num_write: {self.num_write}\n"
                f"MAC Instructions: {self.MAC_instructions}\n"
                f"Sum Instructions: {self.sum_instructions}\n"
                f"Save Instructions: {self.save_instructions}\n"
                f"registres used: {self.registers}\n"
                f"time: {self.time}")

def metrics_counter_dec(func):
    """decorator for colecting metrics. Goes around write
    Args:
        func (write functions): collects args and extracts time consideration
    """
    def wrapper(instruction_type, *args, **kwargs):
        if instruction_type == 'MAC':
            computer_id, *_, size = args
            if computer_id == 1:
                metrics_counter.increment('time', amount=size * PHOTONIC_TIME_MULTIPLIER)
        elif instruction_type == 'sum':
            largest_register = max(args)
            metrics_counter.increment('register', function = max, amount=largest_register)

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
    if isinstance(node, dict):
        for key in node:
            if contains(node[key], val):
                return True
    else:
        if val in node:
            return True
    return False


def get_shape_index(node):
    '''abstracts how to get index's'''
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
    is_remainder = False
    if remainder:
        is_remainder = True
    start = 0
    end = batch_size
    print(temp, batch_size, remainder)
    for i in range(num_batches):
        if remainder == 0 and is_remainder:
            batch_size -= 1
            end -= 1
        if temp < batch_size:
            end = start + temp
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
    assert num_photonic_hardware > 0, 'num_photonic_hardware was less than 1'
    while True:
        for i in range(1, num_photonic_hardware+1, 1):
            yield i


@metrics_counter_dec
def write_instruction(instruction_type, *args):
    """Write abstraction to be called elswhere. Also where metrics are caught
    Args:
        instruction_type (int): name of instruction.
    """
    instruction_format = { # name: (fsrt, (args) )
        'null': ("N: [null] {order}\n",
                ('order',)),
        'other': ("E: [other] {order}\n",
                ('order',)),
        'dense': ("   [relu/MAC][{order}]\n",
                ('order',)),
        'sum': ("E: sum: a{a1}, [a{a2}: a{a3}]\n",
                ('a1', 'a2', 'a3')),
        'MAC': ("P{computerID}: MAC: {write}, SRAM: {v1}, DRAM: {v2}\n",
                ('computerID', 'write', 'v1', 'v2', 'size')),
        # 'load_vector': ("E: load vector: a{a1}, {matrix}[{matrix_index}]{batch}\n",
        #         ('a1', 'matrix', 'matrix_index', 'batch')),
        'save': ("E: save:{write}, {read}\n",
                ('write', 'read'))
    }

    if instruction_type in instruction_format and WRITE_TO_FILE:
        format_string, format_args = instruction_format[instruction_type]
        argument_dict = dict(zip(format_args, args)) # dictionary mapping term name to value
        if instruction_type in ['null', 'other', 'dense']:
            argument_dict['order'] = raw_json['nodes'][args[0]]
        parsed_txt.write(format_string.format(**argument_dict))

def opt_strat(node, optimization):

    def _batching_rows(num_photon_hardware, batch_gen, row):
        # generator object seperating each vector slice
        largest_batch = 0
        for batch in batch_gen:
            size = batch[1]-batch[0]
            largest_batch = max(largest_batch, size)
            batch = f"[{batch[0]}:{batch[1]}]"
            v1 = f"{vector}{batch}"
            v2 = f"{matrix}[{row}]{batch}"
            photonic_hardware_id = next(P_computer_num_gen)
            write_instruction('MAC',photonic_hardware_id,
                                f'a{photonic_hardware_id-1}', v1, v2, size)
        write_instruction("sum", 0, 0, num_photon_hardware)
        write = f'[1:{matrix[0]}][{row}]'
        write_instruction("save",write, 'a0')

    def task_parallel(num_photon_hardware, node):
        # start prioritize - one hardware per matrix row
        for matrix_row in range(matrix[0]):
            write = f'[1:{matrix[0]}][{matrix_row}]'
            v1 = f'{vector}'
            v2 = f'{matrix}[{matrix_row}]'
            size = vector[1]
            write_instruction('MAC',next(P_computer_num_gen), write, v1, v2, size)

    def data_parrellel(num_photon_hardware, node):
        # Finish prioritize - all hardware per matrix row.
        for matrix_row in range(matrix[0]):
            # generator object seperating each vector slice
            batch_gen = batch_vector(matrix[1], num_photon_hardware)
            _batching_rows(num_photon_hardware, batch_gen, matrix_row)
        adder_time = math.log2(num_photon_hardware)* ELECTRONIC_TIME_MULTIPLIER
        metrics_counter.increment('time', amount = adder_time)

    def dynamic_parallel(num_photon_hardware, node):
        """ Task_para untill oversipll. Then data paralelize """
        rows_left = matrix[0]
        while rows_left >= num_photon_hardware:
            for _ in range(num_photon_hardware):
                write = f'[1:{matrix[0]}][{matrix[0] - rows_left}]'
                v1 = f'{vector}'
                v2 = f'{matrix}[{matrix[0] - rows_left}]'
                size = vector[1]
                write_instruction('MAC',next(P_computer_num_gen), write, v1, v2, size)
                rows_left -= 1
        if rows_left:
            larger_batch_size = math.ceil(num_photon_hardware / rows_left)
            small_batch_size = math.floor(num_photon_hardware / rows_left)
            rows_larger = num_photon_hardware % rows_left
            while rows_larger:
                batch_gen = batch_vector(matrix[1], larger_batch_size)
                _batching_rows(num_photon_hardware, batch_gen, matrix[0] - rows_left)
                rows_larger -= 1
                rows_left -= 1

            # for _ in range(rows_left - (num_photon_hardware % rows_left)):
            while rows_left:
                batch_gen = batch_vector(matrix[1], small_batch_size)
                _batching_rows(num_photon_hardware, batch_gen, matrix[0] - rows_left)
                rows_larger -= 1
                rows_left -= 1

            adder_time = math.log2(num_photon_hardware/
                                    larger_batch_size) * ELECTRONIC_TIME_MULTIPLIER
            metrics_counter.increment('time', amount = adder_time)

    def memory_limp():
        """ data_parallel untill memory limit, then task parallel """
        pass


    optimization_algs = {'task_para': task_parallel,
                        'data_para': data_parrellel,
                        'dynamic_para': dynamic_parallel}

    vector_index, matrix_index = get_shape_index(node)
    vector = raw_json['attrs']['shape'][1][vector_index]
    matrix = raw_json['attrs']['shape'][1][matrix_index]

    if WRITE_TO_FILE:
        parsed_txt.write(f"   [read] SRAM:{vector_index}, DRAM:{matrix_index}\n") # indicies
        parsed_txt.write(f"   [MAC] {vector} x {matrix}\n")

    P_computer_num_gen = looper(NUM_PHOTON_HARDWARE)
    optimization_algs[optimization](NUM_PHOTON_HARDWARE, node)

    if WRITE_TO_FILE: parsed_txt.write(f"E: [relu] [1:{matrix[0]}]\n")

#endregion


# Loop over Nodes
def main_loop(num_photon_hardware, optimization = ""):
    """Loops over Relay IR Json nodes and catagorizes them
    Args:
        num_photon_hardware (int ): numer of hardware to simulate
        optimization (str, optional): optimization to run for generation. Defaults to "".
    """
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
# creates the write file in write mode append ('a') mode also exists
parsed_txt = open(output_file_path, "w")
#endregion


GRAPH = True
WRITE = 'dynamic_para'
PHOTONIC_TIME_MULTIPLIER = 10**-10 # photonic is 10 Ghz
ELECTRONIC_TIME_MULTIPLIER = 10**-8 # electronic is .1Ghz
NUM_PHOTON_HARDWARE = 10_000

opts = ['task_para', 'data_para', 'dynamic_para']

if GRAPH is False:
    num_read_plot = []
    num_write_plot = []
    MAC_instructions_plot = []
    sum_instructions_plot = []
    save_instructions_plot = []
    registers_plot = []
    time_plot = []
    num_photonic_hardware_plot = []

    WRITE_TO_FILE = False

    for opt in opts:
        if opt == WRITE:
            WRITE_TO_FILE = True
        metrics_counter = MetricsCounter(opt)
        main_loop(NUM_PHOTON_HARDWARE, optimization = opt)
        print(metrics_counter)
        print('\n')
        WRITE_TO_FILE = False

else:
    # region plotting
    optimizations = ['task_para', 'data_para', 'dynamic_para']
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    for opt in optimizations:
        num_read_plot = []
        num_write_plot = []
        MAC_instructions_plot = []
        sum_instructions_plot = []
        save_instructions_plot = []
        registers_plot = []
        time_plot = []
        num_photonic_hardware_plot = []

        WRITE_TO_FILE = False

        for NUM_PHOTON_HARDWARE in range(1, 1000, 50): # 1000 photonic hardware
            metrics_counter = MetricsCounter(opt)
            num_photonic_hardware_plot.append(NUM_PHOTON_HARDWARE)
            main_loop(NUM_PHOTON_HARDWARE, optimization = opt)

        # color = 'tab:red'
        ax1.set_xlabel('time (s)')
        ax1.set_ylabel('other')
        # ax1.plot(num_photonic_hardware_plot,MAC_instructions_plot, label = f"MACs: {opt}")
        # ax1.plot(num_photonic_hardware_plot,sum_instructions_plot, label = f"ADDs: {opt}")
        # ax1.plot(num_photonic_hardware_plot,save_instructions_plot, label = f"SAVEs: {opt}")
        # ax1.plot(num_photonic_hardware_plot,registers_plot, label = f"REGISTERs: {opt}")
        ax1.tick_params(axis='y')

        # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        # color = 'tab:blue'
        ax2.set_ylabel('Time')  # we already handled the x-label with ax1
        ax2.plot(num_photonic_hardware_plot,time_plot, label = f"time: {opt}")
        ax2.tick_params(axis='y')


    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(lines + lines2, labels + labels2, loc=(.6,.8 ))


    plt.title('all optimization (overlap time)')
    plt.savefig('plot.png')

    #endregion
