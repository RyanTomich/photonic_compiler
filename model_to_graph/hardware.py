import math

# region constants and helpers
MEMORY_TRANSFER_WIDTH = 32  # bits per cycle
DAC_ADC_DELAY = 1 * 10**-8  # 10 nano-seconds

BITS_PER_NUM = 32

# Power
PICO_JOULE = 10**-12

JOULE_PER_CYCLE = 1 * PICO_JOULE

DRAM_READ = 160 * PICO_JOULE
DRAM_WRITE = 160 * PICO_JOULE
HBM_READ = 40 * PICO_JOULE
HBM_WRITE = 40 * PICO_JOULE
SRAM_READ = 12 * PICO_JOULE
SRAM_WRITE = 12 * PICO_JOULE
LOCAL_READ = 1 * PICO_JOULE
LOCAL_WRITE = 1 * PICO_JOULE
PHU_MAC = 0.04 * PICO_JOULE

DAC_POWER = 3.18 * PICO_JOULE
ADC_POWER = 1.6 * PICO_JOULE


# Timtions
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
    return ten_elm(o[0])

def standard_energy(i, o):
    return JOULE_PER_CYCLE

def constnat(c):
    return lambda i, o: {cpu: c}


def elm_const(matrix, const=1):
    return ten_elm(matrix) * const


# endregion


# region Hardware
class HardwareAlgorithm:
    def __init__(self, opp, cost): #hardware: (time, energy)
        self.opp = opp
        self.cost = cost

    def time_cost(self, i, o):
        return sum(hardware.clock_period * cost_tup[0] for hardware, cost_tup in self.cost.items())

    def energy_cost(self, i, o):
        return sum(cost_tup[1] for hardware, cost_tup in self.cost.items())


class Hardware:
    _universal_hardware_ID = 0
    def __init__(self, clock_speed):
        self.clock_speed = clock_speed
        self.clock_period = 1 / clock_speed
        Hardware._universal_hardware_ID += 1

    def __hash__(self):
        return hash(Hardware._universal_hardware_ID)


class PHU(Hardware):
    def __init__(self, clock_speed, num_cores, num_multiplex):
        super().__init__(clock_speed)
        self.num_numtiplex = num_multiplex
        self.num_cores = num_cores
        self.mac_energy = 0.04 * PICO_JOULE

    def _phu_matmul_task_para_cycles(self, i, o):
        num_dot_products = ten_elm(o[0])
        length_dot_products = i[0][-1]
        phu_cycles = (
            math.ceil(math.ceil(num_dot_products / self.num_numtiplex) / self.num_cores)
            * length_dot_products
        )
        return {"PHU": phu_cycles}

    def _phu_matmul_task_para_energy(self, i, o):
        num_dot_products = ten_elm(o[0])
        length_dot_products = i[0][-1]
        return num_dot_products * length_dot_products * self.mac_energy


class CPU(Hardware):
    def __init__(self, clock_speed, num_cores):
        super().__init__(clock_speed)
        self.num_cores = num_cores
        self.mac_energy = 0.1 * PICO_JOULE

    def _cpu_matmul_cycles(self, i, o):
        num_dot_products = ten_elm(o[0])
        length_dot_products = i[0][-1]
        ops_per_mac = 2  # multiplicaiton and addition
        return {"CPU": num_dot_products * length_dot_products * ops_per_mac}

    def _cpu_matmul_energy(self, i, o):
        num_dot_products = ten_elm(o[0])
        length_dot_products = i[0][-1]
        return num_dot_products * length_dot_products * self.mac_energy


class GPU(Hardware):
    def __init__(self, clock_speed):
        super().__init__(clock_speed)
        self.mac_energy = 0.1 * PICO_JOULE


class HBM(Hardware):
    def __init__(self, clock_speed):
        super().__init__(clock_speed)


class SRAM(Hardware):
    def __init__(self, clock_speed):
        super().__init__(clock_speed)


# endregion

cpu = CPU(10**8, 1)
phu = PHU(10**10, 64, 20)

hardware_algs = {
"add": HardwareAlgorithm("add", {cpu: (all_elm, standard_energy)} ),
"subtract": HardwareAlgorithm("subtract", {cpu: (all_elm, standard_energy)} ),
"multiply": HardwareAlgorithm("multiply", {cpu: (all_elm, standard_energy)} ),
"divide": HardwareAlgorithm("divide", {cpu: (all_elm, standard_energy)} ),
"sqrt": HardwareAlgorithm("sqrt", {cpu: (all_elm, standard_energy)} ),
"rsqrt": HardwareAlgorithm("rsqrt", {cpu: (all_elm, standard_energy)} ),
"relu": HardwareAlgorithm("relu", {cpu: (all_elm, standard_energy)} ),
"tanh": HardwareAlgorithm("tanh", {cpu: (lambda i, o: ten_elm(o[0]) * 4, standard_energy) } ),

"power": HardwareAlgorithm("power", {cpu: (all_elm, standard_energy) }),
"transpose": HardwareAlgorithm("transpose", {cpu: (all_elm, standard_energy)} ),
"nop": HardwareAlgorithm("nop", {cpu: (constnat(1), standard_energy)} ),
"less": HardwareAlgorithm("less", {cpu: (constnat(1), standard_energy)} ),
"take": HardwareAlgorithm("take", {cpu: (constnat(1), standard_energy)} ),
"mean": HardwareAlgorithm(
    "mean", {cpu: (lambda i, o: (i[0][-1] + 1) * i[0][-2], standard_energy) }),
"softmax": HardwareAlgorithm(
    "softmax", {cpu: (lambda i, o: ten_elm(o[0]) * 6, standard_energy) }
),
"matmul": HardwareAlgorithm(
    "matmul", {cpu: (cpu._cpu_matmul_cycles, cpu._cpu_matmul_energy)}
),
"dense": HardwareAlgorithm(
    "dense", {cpu: (cpu._cpu_matmul_cycles, cpu._cpu_matmul_energy)}
),
"pack": HardwareAlgorithm(
    "pack", {cpu: (cpu._cpu_matmul_cycles, cpu._cpu_matmul_energy)}
),
"where": HardwareAlgorithm("where", {cpu: (constnat(1), standard_energy) }),
"erf": HardwareAlgorithm("erf",  {cpu: (constnat(1), standard_energy) }),
"task_para_matmul_phu": HardwareAlgorithm(
    "matmul",
    {phu:
    (phu._phu_matmul_task_para_cycles,
    phu._phu_matmul_task_para_energy)
    }
),
"task_para_dense_phu": HardwareAlgorithm(
    "dense",
    {phu:
    (phu._phu_matmul_task_para_cycles,
    phu._phu_matmul_task_para_energy)
    }
),
"task_para_pack_phu": HardwareAlgorithm(
    "pack",
    {phu:
    (phu._phu_matmul_task_para_cycles,
    phu._phu_matmul_task_para_energy)
    }
),
"dot_prod_phu": HardwareAlgorithm(
    "dot_prod",
    {phu:
    (lambda i, o: i[0][-1],
    lambda i, o: i[0][-1] * phu.mac_energy)
    }
),

}
