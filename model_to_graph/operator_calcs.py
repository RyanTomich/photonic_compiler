"""
simulations constants and definitions
"""

import math
import numpy as np


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


def phu_matmul_task_para_time(i, o):
    num_dot_products = ten_elm(i[0]) / i[0][-1]
    length_dot_products = i[0][-1]
    phu_cycles = (
        math.ceil(math.ceil(num_dot_products / PHU_MULTIPLEX) / PHU_CORES)
        * length_dot_products
    )
    return {"PHU": phu_cycles}


def phu_matmul_dynamic_para_time(i, o):  # TODO
    cores_per_partition = int(PHU_CORES)
    return {"PHU": ten_elm(i[0]) * i[1][-2] / cores_per_partition}


def func():
    print("placeholder")


def hw_time_intercon(hardware, bits):
    if any(i in ["start", "SRAM"] for i in hardware):
        return hw_time_intercon_dict[hardware](bits)
    else:  # must go through SRAM
        return sum(
            hw_time_intercon_dict[(to_sram, "SRAM")](bits) for to_sram in hardware
        )


def hw_energy_intercon(hardware, bits):
    if any(i in ["start", "SRAM"] for i in hardware):
        return hw_energy_intercon_dict[hardware](bits)
    # must go through SRAM
    return sum(hw_energy_intercon_dict[(to_sram, "SRAM")](bits) for to_sram in hardware)


# Constants
NODE_COUNT = 0

CPU_CLOCK_SPEED = 10**8  # .1Ghz
PHU_CLOCK_SPEED = 10**10  # 10 Ghz

PHU_CORES = 64
PHU_MULTIPLEX = 20
CPU_CORES = 1

DRAM_SRAM_WIDTH = 256  # bits per cycle
SRAM_OVERHEAD = 5  # electronic cycles
MODULATOR_CONST = 1 / PHU_CLOCK_SPEED  # per bit time of electronic-photonic conversion
BITS_PER_NUM = 8

J_PER_BIT = 10**-12  # 1 pico-jule

cycle_to_time_funcs = {
    "CPU": lambda x: x / CPU_CLOCK_SPEED,
    "PHU": lambda x: x / PHU_CLOCK_SPEED,
    "DRAM": lambda x: x / CPU_CLOCK_SPEED,
}

node_value_selection = {
    "time": lambda node: node.time_cost,
    "energy": lambda node: node.energy_cost,
}
edge_value_selection = {
    "time": hw_time_intercon,
    "energy": hw_energy_intercon,
}


class HardwareAlgorithm:
    def __init__(
        self,
        opp,
        hardware,
        function,
        cycle_function,
        energy_function=lambda i, o: 10 * J_PER_BIT,
    ):
        self.opp = opp
        self.hardware = hardware
        self.func = function
        self.cycle_function = cycle_function
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
        "matmul", "CPU", func, lambda i, o: {"CPU": ten_elm(i[0]) * i[1][-2]}
    ),
    "dense": HardwareAlgorithm(
        "dense", "CPU", func, lambda i, o: {"CPU": ten_elm(i[0]) * i[1][-2]}
    ),
    "pack": HardwareAlgorithm(
        "pack", "CPU", func, lambda i, o: {"CPU": ten_elm(i[0]) * i[1][-2]}
    ),
    "where": HardwareAlgorithm("where", "CPU", func, constnat(1)),
    "erf": HardwareAlgorithm("erf", "CPU", func, constnat(1)),
    "task_para_matmul_phu": HardwareAlgorithm(
        "matmul", "PHU", func, phu_matmul_task_para_time, lambda i, o: 5 * J_PER_BIT
    ),
    "task_para_dense_phu": HardwareAlgorithm(
        "dense", "PHU", func, phu_matmul_task_para_time, lambda i, o: 5 * J_PER_BIT
    ),
    "task_para_pack_phu": HardwareAlgorithm(
        "pack", "PHU", func, phu_matmul_task_para_time, lambda i, o: 5 * J_PER_BIT
    ),
    "dynamic_para_matmul_phu": HardwareAlgorithm(
        "matmul", "PHU", func, phu_matmul_dynamic_para_time
    ),
    "dynamic_para_dense_phu": HardwareAlgorithm(
        "dense", "PHU", func, phu_matmul_dynamic_para_time
    ),
    "dynamic_para_pack_phu": HardwareAlgorithm(
        "pack", "PHU", func, phu_matmul_dynamic_para_time
    ),
    "get_dram": HardwareAlgorithm(
        "null",
        "DRAM",
        func,
        lambda i, o: {"DRAM": ten_elm(i) * BITS_PER_NUM / DRAM_SRAM_WIDTH},
    ),
    "start": HardwareAlgorithm("start", "start", func, constnat(1)),
    "dot_prod_phu": HardwareAlgorithm(
        "dot_prod", "PHU", func, lambda i, o: {"PHU": i[0][-1]}
    ),
}


hw_time_intercon_dict = {  # size to time between hardware
    # ("DRAM", "SRAM"): lambda x: 10 / CPU_CLOCK_SPEED,
    # ("CPU", "SRAM"): lambda x: SRAM_OVERHEAD
    # / CPU_CLOCK_SPEED,  # SRAM clock cycle overhead
    # ("PHU", "SRAM"): lambda x: SRAM_OVERHEAD / CPU_CLOCK_SPEED + x * MODULATOR_CONST,
    ("DRAM", "SRAM"): lambda x: 1 / CPU_CLOCK_SPEED,
    ("CPU", "SRAM"): lambda x: 1 / CPU_CLOCK_SPEED,
    ("PHU", "SRAM"): lambda x: 1 / CPU_CLOCK_SPEED,
    ("SRAM", "SRAM"): lambda x: np.inf,
    ("CPU", "start"): lambda x: 0,
    ("PHU", "start"): lambda x: 0,
    ("start", "SRAM"): lambda x: 0,
    ("start", "start"): lambda x: np.inf,
}

hw_energy_intercon_dict = {  # size to energy between hardware
    ("DRAM", "SRAM"): lambda x: x * J_PER_BIT,
    ("CPU", "SRAM"): lambda x: x * J_PER_BIT,
    ("PHU", "SRAM"): lambda x: x * 3 * J_PER_BIT,  # TODO
    ("CPU", "start"): lambda x: 0,
    ("PHU", "start"): lambda x: 0,
    ("start", "SRAM"): lambda x: 0,
}
