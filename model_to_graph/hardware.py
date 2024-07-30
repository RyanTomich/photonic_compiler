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
GPU_MAC = 0.1 * PICO_JOULE
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
    return {"CPU": ten_elm(o[0])}


def constnat(c):
    return lambda i, o: {"CPU": c}


def elm_const(matrix, const=1):
    return ten_elm(matrix) * const


# endregion


# region Hardware
class HardwareAlgorithm:
    def __init__(
        self,
        opp,
        hardware,
        cycle_function,
        energy_funcion=lambda i, o: JOULE_PER_CYCLE,
    ):
        self.opp = opp
        self.hardware = hardware
        self.cycle_function = cycle_function

        # default value untill energy for all other algs
        self.energy_funcion = energy_funcion

    def time_cost(self, i, o):
        total = 0
        for hardware, cycles in self.cycle_function(i, o).items():
            total += cycle_to_tims[hardware](cycles)
        return total

    def energy_cost(self, i, o):
        return self.energy_funcion(i, o)


class Hardware:
    def __init__(self, clock_speed):
        self.clock_speed = clock_speed
        self.clock_period = 1 / clock_speed


class PHU(Hardware):
    def __init__(self, clock_speed, num_cores, num_multiplex):
        super().__init__(clock_speed)
        self.num_numtiplex = num_multiplex
        self.num_cores = num_cores
        self.hardware_algs = {
            "task_para_matmul_phu": HardwareAlgorithm(
                "matmul",
                "PHU",
                self._phu_matmul_task_para_cycles,
                self._phu_matmul_task_para_energy,
            ),
            "task_para_dense_phu": HardwareAlgorithm(
                "dense",
                "PHU",
                self._phu_matmul_task_para_cycles,
                self._phu_matmul_task_para_energy,
            ),
            "task_para_pack_phu": HardwareAlgorithm(
                "pack",
                "PHU",
                self._phu_matmul_task_para_cycles,
                self._phu_matmul_task_para_energy,
            ),
            "dot_prod_phu": HardwareAlgorithm(
                "dot_prod",
                "PHU",
                lambda i, o: {"PHU": i[0][-1]},
                lambda i, o: i[0][-1] * PHU_MAC,
            ),
        }

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
        return num_dot_products * length_dot_products * PHU_MAC


class CPU(Hardware):
    def __init__(self, clock_speed, num_cores):
        super().__init__(clock_speed)
        self.num_cores = num_cores
        self.hardware_algs = {
            "add": HardwareAlgorithm("add", "CPU", all_elm),
            "subtract": HardwareAlgorithm("subtract", "CPU", all_elm),
            "multiply": HardwareAlgorithm("multiply", "CPU", all_elm),
            "divide": HardwareAlgorithm("divide", "CPU", all_elm),
            "sqrt": HardwareAlgorithm("sqrt", "CPU", all_elm),
            "rsqrt": HardwareAlgorithm("rsqrt", "CPU", all_elm),
            "relu": HardwareAlgorithm("relu", "CPU", all_elm),
            "tanh": HardwareAlgorithm(
                "tanh", "CPU", lambda i, o: {"CPU": ten_elm(o[0]) * 4}
            ),
            "power": HardwareAlgorithm("power", "CPU", all_elm),
            "transpose": HardwareAlgorithm("transpose", "CPU", all_elm),
            "nop": HardwareAlgorithm("nop", "CPU", constnat(1)),
            "less": HardwareAlgorithm("less", "CPU", constnat(1)),
            "take": HardwareAlgorithm("take", "CPU", constnat(1)),
            "mean": HardwareAlgorithm(
                "mean", "CPU", lambda i, o: {"CPU": (i[0][-1] + 1) * i[0][-2]}
            ),
            "softmax": HardwareAlgorithm(
                "softmax", "CPU", lambda i, o: {"CPU": ten_elm(o[0]) * 6}
            ),
            "matmul": HardwareAlgorithm(
                "matmul", "CPU", self._cpu_matmul_cycles, self._cpu_matmul_energy
            ),
            "dense": HardwareAlgorithm(
                "dense", "CPU", self._cpu_matmul_cycles, self._cpu_matmul_energy
            ),
            "pack": HardwareAlgorithm(
                "pack", "CPU", self._cpu_matmul_cycles, self._cpu_matmul_energy
            ),
            "where": HardwareAlgorithm("where", "CPU", constnat(1)),
            "erf": HardwareAlgorithm("erf", "CPU", constnat(1)),
        }

    def _cpu_matmul_cycles(self, i, o):
        num_dot_products = ten_elm(o[0])
        length_dot_products = i[0][-1]
        ops_per_mac = 2  # multiplicaiton and addition
        return {"CPU": num_dot_products * length_dot_products * ops_per_mac}

    def _cpu_matmul_energy(self, i, o):
        num_dot_products = ten_elm(o[0])
        length_dot_products = i[0][-1]
        return num_dot_products * length_dot_products * GPU_MAC


class GPU(Hardware):
    def __init__(self, clock_speed):
        super().__init__(clock_speed)


class HBM(Hardware):
    def __init__(self, clock_speed):
        super().__init__(clock_speed)


class SRAM(Hardware):
    def __init__(self, clock_speed):
        super().__init__(clock_speed)


# endregion

CPU(10**8, 1)
PHU(10**10, 64, 20)
