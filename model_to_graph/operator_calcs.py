import numpy as np


# Time Estamates
def phu_matmul_task_para_time(i, o):
    cores_per_partition = int(PHU_CORES)
    return {"PHU": ten_elm(i[0]) * i[1][-2] / cores_per_partition}


def phu_matmul_dynamic_para_time(i, o):
    cores_per_partition = int(PHU_CORES)
    return {"PHU": ten_elm(i[0]) * i[1][-2] / cores_per_partition}


def creat_available_hardware(hardware_dict):
    hardware = {}
    for hw, count in hardware_dict.items():
        in_dict = {}
        for i in range(count):
            in_dict[f"{hw}{i}"] = 0
        hardware[hw] = in_dict
    return hardware


def initilize_hardware():
    global available_hardware
    available_hardware = creat_available_hardware(
        {"CPU": CPU_CORES, "PHU": PHU_CORES, "SRAM": 1}
    )


def ten_elm(tensor_shape):
    ans = 1
    for dimention in tensor_shape:
        ans *= dimention
    return ans


def elm_const(matrix, const=1):
    return ten_elm(matrix) * const


def cycle_to_s(cost):
    if not isinstance(cost, dict):
        return cost
    total = 0
    for hardware, cycles in cost.items():
        total += cycle_to_time_funcs[hardware](cycles)
    return total


def func():
    print("placeholder")


# Constants
NODE_COUNT = 0

CPU_CLOCK_SPEED = 10**8  # .1Ghz
PHU_CLOCK_SPEED = 10**10  # 10 Ghz

PHU_CORES = 64
PHU_MULTIPLEX = 32
CPU_CORES = 8

DRAM_SRAM_WIDTH = 256  # bits per cycle
SRAM_OVERHEAD = 5  # electronic cycles
MODULATOR_CONST = 1 / PHU_CLOCK_SPEED  # per bit time of electronic-photonic conversion
BITS_PER_NUM = 8

cycle_to_time_funcs = {
    "CPU": lambda x: x / CPU_CLOCK_SPEED,
    "PHU": lambda x: x / PHU_CLOCK_SPEED,
    "DRAM": lambda x: x / CPU_CLOCK_SPEED,
}


def all_elm(i, o):
    return {"CPU": ten_elm(o[0])}


def constnat(c):
    return lambda i, o: {"CPU": c}


hardware_algs = {  # name: (opp, hardware, func, cycles)
    "add": ("add", "CPU", func, all_elm),
    "subtract": ("subtract", "CPU", func, all_elm),
    "multiply": ("multiply", "CPU", func, all_elm),
    "divide": ("divide", "CPU", func, all_elm),
    "sqrt": ("sqrt", "CPU", func, all_elm),
    "rsqrt": ("rsqrt", "CPU", func, all_elm),
    "relu": ("relu", "CPU", func, all_elm),
    "tanh": ("tanh", "CPU", func, lambda i, o: {"CPU": ten_elm(o[0]) * 4}),
    "power": ("power", "CPU", func, all_elm),
    "transpose": ("transpose", "CPU", func, all_elm),
    "nop": ("nop", "CPU", func, constnat(1)),
    "less": ("less", "CPU", func, constnat(1)),
    "take": ("take", "CPU", func, constnat(1)),
    "split": ("split", "SRAM", func, constnat(3)),
    "mean": (
        "mean",
        "CPU",
        func,
        lambda i, o: {"CPU": (i[0][-1] + 1) * i[0][-2]},
    ),
    "softmax": (
        "softmax",
        "CPU",
        func,
        lambda i, o: {"CPU": ten_elm(o[0]) * 6},
    ),
    "matmul": (
        "matmul",
        "CPU",
        func,
        lambda i, o: {"CPU": ten_elm(i[0]) * i[1][-2] * 2},
    ),
    "dense": (
        "dense",
        "CPU",
        func,
        lambda i, o: {"CPU": ten_elm(i[0]) * i[1][-2] * 2},
    ),
    "pack": (
        "pack",
        "CPU",
        func,
        lambda i, o: {"CPU": ten_elm(i[0]) * i[1][-2] * 2},
    ),
    "where": ("where", "CPU", func, constnat(1)),
    "erf": ("erf", "CPU", func, constnat(1)),  # Bert cumulative distribution function??
    "task_para_matmul_phu": ("matmul", "PHU", func, phu_matmul_task_para_time),
    "dynamic_para_matmul_phu": ("matmul", "PHU", func, phu_matmul_dynamic_para_time),
    "task_para_dense_phu": ("dense", "PHU", func, phu_matmul_task_para_time),
    "dynamic_para_dense_phu": ("dense", "PHU", func, phu_matmul_dynamic_para_time),
    "task_para_pack_phu": ("pack", "PHU", func, phu_matmul_task_para_time),
    "dynamic_para_pack_phu": ("pack", "PHU", func, phu_matmul_dynamic_para_time),
    "get_dram": (
        "null",
        "DRAM",
        func,
        lambda i, o: {"DRAM": ten_elm(i) * BITS_PER_NUM / DRAM_SRAM_WIDTH},
    ),
    "start": (
        "start",
        "start",
        func,
        constnat(1),
    ),  # Here for mock start nodes in optimization.
    "dot_prod_phu": ("dot_prod", "PHU", func, lambda i, o: {"PHU": i[0][-1]}),
}


def hw_intercon(hardware, bits):
    if "start" in hardware:
        return 0
    elif "SRAM" in hardware:
        return hw_intercon_dict[hardware](bits)
    else:
        total = 0
        for to_sram in hardware:
            total += hw_intercon_dict[(to_sram, "SRAM")](bits)
        return total


hw_intercon_dict = {
    ("DRAM", "SRAM"): lambda x: 10 / CPU_CLOCK_SPEED,
    ("CPU", "SRAM"): lambda x: SRAM_OVERHEAD
    / CPU_CLOCK_SPEED,  # SRAM clock cycle overhead
    ("PHU", "SRAM"): lambda x: SRAM_OVERHEAD / CPU_CLOCK_SPEED + x * MODULATOR_CONST,
    ("SRAM", "SRAM"): lambda x: np.inf,
    # mock start nodes in optimization.
    ("CPU", "start"): lambda x: 0,
    ("PHU", "start"): lambda x: 0,
    ("start", "start"): lambda x: np.inf,
}
