"""
simulations constants and definitions
"""

import math
import numpy as np

DEBUG_PRINT = False


# Scheduling
def creat_available_hardware(hardware_dict):
    """Given the machiene hardware, create hardware dict

    Args:
        hardware_dict (dict): hardware type to number of that hardware

    Returns:
        dict: each individual hardware to time (0)
    """
    hardware = {}
    for hw, count in hardware_dict.items():
        in_dict = {}
        for i in range(count):
            in_dict[f"{hw}{i}"] = 0
        hardware[hw] = in_dict
    return hardware


def initilize_hardware():
    """Creates global definition of the hardware type to count.
    This is here to explicitly create hardware once when needed
    """
    global available_hardware
    available_hardware = creat_available_hardware(
        {"CPU": CPU_CORES, "PHU": PHU_CORES, "SRAM": 1}
    )


# Time functions
def ten_elm(tensor_shape):
    """
    Args:
        tensor_shape (lsit or tuple)

    Returns:
        int: number of elements
    """
    ans = 1
    for dimention in tensor_shape:
        ans *= dimention
    return ans


def all_elm(i, o):
    return {"CPU": ten_elm(o[0])}


def constnat(c):
    return lambda i, o: {"CPU": c}


def elm_const(matrix, const=1):
    return ten_elm(matrix) * const


## CPU
def cpu_matmul_time(i, o):
    return {"CPU": ten_elm(i[0]) * i[1][-2]}


def cpu_matmul_energy(i, o):
    return ten_elm(i[0]) * i[1][-2] * GPU_MAC


## PHU
def phu_matmul_task_para_time(i, o):
    num_dot_products = ten_elm(i[0]) / i[0][-1]
    length_dot_products = i[0][-1]
    phu_cycles = (
        math.ceil(math.ceil(num_dot_products / PHU_MULTIPLEX) / PHU_CORES)
        * length_dot_products
    )
    return {"PHU": phu_cycles}


def phu_matmul_task_para_energy(i, o):
    num_dot_products = ten_elm(i[0]) / i[0][-1]
    length_dot_products = i[0][-1]
    return num_dot_products * length_dot_products * PHU_MAC


def phu_matmul_dynamic_para_time(i, o):  # TODO
    cores_per_partition = int(PHU_CORES)
    return {"PHU": ten_elm(i[0]) * i[1][-2] / cores_per_partition}


def func():
    print("placeholder")


def edge_value_selection(
    optimization_variable, hw_connection, num_transfer, bit_transfer
):
    """Calculates cost between hardware. Accounts for trips to and from SRAM

    Args:
        optimization_variable (str): time or energy
        hw_connection (tuple): (hw1, hw2 )
        num_transfer (int): number of number being transfered
        bit_transfer (int): number of bits being transfered

    Returns:
        int: total cost in seconds or jules (depending on optimization_variable)
    """
    if any(i in ["start", "SRAM"] for i in hw_connection):  # one way
        return hw_intercon[hw_connection].get_cost(
            optimization_variable, num_transfer, bit_transfer
        )
    else:  # must go through SRAM
        return sum(
            (
                hw_intercon[(hw_connection[0], "SRAM")].get_cost(
                    optimization_variable, num_transfer, bit_transfer
                ),
                hw_intercon[("SRAM", hw_connection[1])].get_cost(
                    optimization_variable, num_transfer, bit_transfer
                ),
            )
        )


# region ###  Constants ###
NODE_COUNT = 0

# Time
CPU_CLOCK_SPEED = 10**8  # .1Ghz
PHU_CLOCK_SPEED = 10**10  # 10 Ghz

PHU_CORES = 64
PHU_MULTIPLEX = 20
CPU_CORES = 1

DRAM_SRAM_WIDTH = 256  # bits per cycle
SRAM_OVERHEAD = 5  # electronic cycles
MODULATOR_CONST = 1 / PHU_CLOCK_SPEED  # per bit time of electronic-photonic conversion
BITS_PER_NUM = 8

# Power
PICO_JOULE = 10**-12

JOULE_PER_CYCLE = 1 * PICO_JOULE

DRAM_READ = 160 * PICO_JOULE
DRAM_WRITE = 160 * PICO_JOULE
HBM_READ = 40 * PICO_JOULE  # TODO
HBM_WRITE = 40 * PICO_JOULE  # TODO
SRAM_READ = 12 * PICO_JOULE
SRAM_WRITE = 12 * PICO_JOULE
LOCAL_READ = 1 * PICO_JOULE  # TODO
LOCAL_WRITE = 1 * PICO_JOULE  # TODO
GPU_MAC = 0.1 * PICO_JOULE
PHU_MAC = 0.04 * PICO_JOULE

# DAC_POWER = np.inf
# ADC_POWER = np.inf
DAC_POWER = 3.18 * PICO_JOULE
ADC_POWER = 1.6 * PICO_JOULE

# endregion

cycle_to_time_funcs = {
    "CPU": lambda x: x / CPU_CLOCK_SPEED,
    "PHU": lambda x: x / PHU_CLOCK_SPEED,
    "DRAM": lambda x: x / CPU_CLOCK_SPEED,
}

node_value_selection = {
    "time": lambda node: node.time_cost,
    "energy": lambda node: node.energy_cost,
}


class HardwareAlgorithm:
    def __init__(
        self,
        opp,
        hardware,
        function,
        cycle_function,
        energy_function=lambda i, o: JOULE_PER_CYCLE,
    ):
        self.opp = opp
        self.hardware = hardware
        self.func = function
        self.cycle_function = cycle_function

        # default value untill energy for all other algs
        self.energy_function = energy_function

    def cycle_to_s(self, cost: dict) -> int:
        total = 0
        for hardware, cycles in cost.items():
            total += cycle_to_time_funcs[hardware](cycles)
        return total

    def time_cost(self, i, o):
        return self.cycle_to_s(self.cycle_function(i, o))

    def energy_cost(self, i, o):
        return self.energy_function(i, o)


hardware_algs = {
    "add": HardwareAlgorithm("add", "CPU", func, all_elm),
    "subtract": HardwareAlgorithm("subtract", "CPU", func, all_elm),
    "multiply": HardwareAlgorithm("multiply", "CPU", func, all_elm),
    "divide": HardwareAlgorithm("divide", "CPU", func, all_elm),
    "sqrt": HardwareAlgorithm("sqrt", "CPU", func, all_elm),
    "rsqrt": HardwareAlgorithm("rsqrt", "CPU", func, all_elm),
    "relu": HardwareAlgorithm("relu", "CPU", func, all_elm),
    "tanh": HardwareAlgorithm(
        "tanh", "CPU", func, lambda i, o: {"CPU": ten_elm(o[0]) * 4}
    ),
    "power": HardwareAlgorithm("power", "CPU", func, all_elm),
    "transpose": HardwareAlgorithm("transpose", "CPU", func, all_elm),
    "nop": HardwareAlgorithm("nop", "CPU", func, constnat(1)),
    "less": HardwareAlgorithm("less", "CPU", func, constnat(1)),
    "take": HardwareAlgorithm("take", "CPU", func, constnat(1)),
    "split": HardwareAlgorithm("split", "SRAM", func, constnat(1)),
    "mean": HardwareAlgorithm(
        "mean", "CPU", func, lambda i, o: {"CPU": (i[0][-1] + 1) * i[0][-2]}
    ),
    "softmax": HardwareAlgorithm(
        "softmax", "CPU", func, lambda i, o: {"CPU": ten_elm(o[0]) * 6}
    ),
    "matmul": HardwareAlgorithm(
        "matmul", "CPU", func, cpu_matmul_time, cpu_matmul_energy
    ),
    "dense": HardwareAlgorithm(
        "dense", "CPU", func, cpu_matmul_time, cpu_matmul_energy
    ),
    "pack": HardwareAlgorithm("pack", "CPU", func, cpu_matmul_time, cpu_matmul_energy),
    "where": HardwareAlgorithm("where", "CPU", func, constnat(1)),
    "erf": HardwareAlgorithm("erf", "CPU", func, constnat(1)),
    "task_para_matmul_phu": HardwareAlgorithm(
        "matmul", "PHU", func, phu_matmul_task_para_time, phu_matmul_task_para_energy
    ),
    "task_para_dense_phu": HardwareAlgorithm(
        "dense", "PHU", func, phu_matmul_task_para_time, phu_matmul_task_para_energy
    ),
    "task_para_pack_phu": HardwareAlgorithm(
        "pack", "PHU", func, phu_matmul_task_para_time, phu_matmul_task_para_energy
    ),
    "dynamic_para_matmul_phu": HardwareAlgorithm(  # TODO
        "matmul", "PHU", func, phu_matmul_dynamic_para_time, lambda i, o: np.inf
    ),
    "dynamic_para_dense_phu": HardwareAlgorithm(  # TODO
        "dense", "PHU", func, phu_matmul_dynamic_para_time, lambda i, o: np.inf
    ),
    "dynamic_para_pack_phu": HardwareAlgorithm(  # TODO
        "pack", "PHU", func, phu_matmul_dynamic_para_time, lambda i, o: np.inf
    ),
    "get_dram": HardwareAlgorithm(
        "null",
        "DRAM",
        func,
        lambda i, o: {"DRAM": ten_elm(i) * BITS_PER_NUM / DRAM_SRAM_WIDTH},
    ),
    "start": HardwareAlgorithm("start", "start", func, constnat(1)),
    "dot_prod_phu": HardwareAlgorithm(
        "dot_prod",
        "PHU",
        func,
        lambda i, o: {"PHU": i[0][-1]},
        lambda i, o: i[0][-1] * PHU_MAC,
    ),
}


class HardwareConnection:
    def __init__(self, time_cost_per_bit, energy_cost_per_number):
        """create functions for cost

        Args:
            time_cost_func (int): time cost in seconds per bit
            energy_cost_func (int): energy cost in joules per bit
        """
        # self.time_cost_func = lambda n, b: b * time_cost_per_bit
        self.time_cost_func = lambda n, b: 1 / CPU_CLOCK_SPEED
        self.energy_cost_func = lambda n, b: n * energy_cost_per_number

    def get_cost(self, optimization_variable, num_transfer, bit_transfer):
        var_to_func = {
            "time": self.time_cost_func,
            "energy": self.energy_cost_func,
        }

        return var_to_func[optimization_variable](num_transfer, bit_transfer)


hw_intercon = {
    ("DRAM", "SRAM"): HardwareConnection(0, DRAM_READ + SRAM_WRITE),
    ("SRAM", "CPU"): HardwareConnection(0, SRAM_READ),
    ("SRAM", "PHU"): HardwareConnection(0, SRAM_READ + DAC_POWER),
    ("CPU", "SRAM"): HardwareConnection(0, SRAM_WRITE),
    ("PHU", "SRAM"): HardwareConnection(0, ADC_POWER + SRAM_WRITE),
    ("SRAM", "DRAM"): HardwareConnection(0, SRAM_READ + DRAM_WRITE),
    ("start", "CPU"): HardwareConnection(0, 0),
    ("start", "PHU"): HardwareConnection(0, 0),
    ("start", "SRAM"): HardwareConnection(0, 0),
}


#     # ("DRAM", "SRAM"): lambda x: 10 / CPU_CLOCK_SPEED,
#     # ("CPU", "SRAM"): lambda x: SRAM_OVERHEAD
#     # / CPU_CLOCK_SPEED,  # SRAM clock cycle overhead
#     # ("PHU", "SRAM"): lambda x: SRAM_OVERHEAD / CPU_CLOCK_SPEED + x * MODULATOR_CONST,
