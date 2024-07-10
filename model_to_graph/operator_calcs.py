import numpy as np

# region ### Depracated

# run_cpu_cycles = {
#     'add':      all_elm,
#     'subtract': all_elm,
#     'multiply': all_elm,
#     'divide':   all_elm,
#     'sqrt':     all_elm,
#     'rsqrt':    all_elm,
#     'tanh':     lambda i, o: {"CPU": ten_elm(o[0])*4}, #e^x definition
#     'power':    all_elm,
#     'transpose':all_elm,
#     'nop':      constnat(0),
#     'less':     constnat(1),
#     'take':     constnat(1),
#     'split':    constnat(3),
#     'mean':     lambda i, o: {"CPU":(i[0][-1]+1)* i[0][-2]},
#     'softmax':  lambda i, o: {"CPU": ten_elm(o[0])*6},
#     'matmul':   lambda i, o: {"CPU": ten_elm(i[0])*i[1][-2]*2},
#     'dense':    lambda i, o: {"CPU": ten_elm(i[0])*i[1][-2]*2},
#     'pack':     lambda i, o: {"CPU": ten_elm(i[0])*i[1][-2]*2},
#     'where':    constnat(1),
# }

# run_phu_cycles = {
#     'matmul':  lambda i, o: {"PHU": ten_elm(i[0])*i[1][-2]},
#     'dense':  lambda i, o: {"PHU": ten_elm(i[0])*i[1][-2]},
#     'dense':  lambda i, o: {"PHU": ten_elm(i[0])*i[1][-2]},
#     'pack':  lambda i, o: {"PHU": ten_elm(i[0])*i[1][-2]},
# }

# cycle_funcions = {"run_cpu": run_cpu_cycles, "run_phu": run_phu_cycles}


# def opp_time_func(opp, input_shapes, output_shapes, hardware_config):
#     '''
#     get the total clock cycles for each hardware choice
#     opp(srt)
#     input_shapes(list of list)
#     output_shapes(list of list)
#     run_hardware[str]: which hardware to run computation.

#     matriceis representing cost of hardware choice
#     ["CPU", #, # ...]

#     [
#         ["CPU", #, # ...]
#         ["PHU", #, # ...]
#     ]
#     '''
#     opp_cycle_dict = cycle_funcions[hardware_config]

#     if opp in opp_cycle_dict:
#         cycles_dict = opp_cycle_dict[opp](input_shapes, output_shapes)
#         time_total = 0
#         for hardware, cycles in cycles_dict.items():
#             time_total += cycle_to_time_funcs[hardware](cycles)
#         return time_total

#     return np.inf

# endregion

# Time Estamates
def phu_matmul_task_para_time(i,o):
    cores_per_partition = int(PHU_CORES)
    return {"PHU": ten_elm(i[0]) * i[1][-2] / cores_per_partition}

def phu_matmul_dynamic_para_time(i,o):
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
    available_hardware = creat_available_hardware({"CPU": CPU_CORES, "PHU": PHU_PARTITIONS})


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

PHU_CORES = 32
PHU_MULTIPLEX = 20
CPU_CORES = 8
PHU_PARTITIONS = 1

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
    "split": ("split", "CPU", func, constnat(3)),
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

    "task_para_matmul_phu": ("matmul","PHU",func,phu_matmul_task_para_time),
    "dynamic_para_matmul_phu": ("matmul","PHU",func,phu_matmul_dynamic_para_time),
    "task_para_dense_phu": ("dense", "PHU", func, phu_matmul_task_para_time),
    "dynamic_para_dense_phu": ("dense", "PHU", func, phu_matmul_dynamic_para_time),
    "task_para_pack_phu": ("pack", "PHU", func, phu_matmul_task_para_time),
    "dynamic_para_pack_phu": ("pack", "PHU", func, phu_matmul_dynamic_para_time),

    "get_dram": (
        "null",
        "SRAM",
        func,
        lambda i, o: {"DRAM": ten_elm(i) * BITS_PER_NUM / DRAM_SRAM_WIDTH},
    ),
    "start": (
        "start",
        "start",
        func,
        constnat(1),
    ),  # Here for mock start nodes in optimization.

    "ghost": (
        "ghost",
        "start",
        func,
        constnat(0),
    ),

    'dot_prod_phu': ('dot_prod', 'PHU', func, lambda i, o: {'PHU': i[0][-1]}),
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
    ("CPU", "SRAM"): lambda x: SRAM_OVERHEAD
    / CPU_CLOCK_SPEED,  # SRAM clock cycle overhead
    ("PHU", "SRAM"): lambda x: SRAM_OVERHEAD / CPU_CLOCK_SPEED + x * MODULATOR_CONST,
    ("SRAM", "SRAM"): lambda x: np.inf,
    # Here for mock start nodes in optimization.
    ("CPU", "start"): lambda x: 0,  # DRAM to SRAM
    ("PHU", "start"): lambda x: 0,
    ("start", "start"): lambda x: np.inf,
}
