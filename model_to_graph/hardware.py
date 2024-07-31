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


# time and dimentions
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
def initilize_hardware(hardware):
    """creates intercon functions

    Args:
        hw (list): list of hardware instances of the system
    """
    sram = SRAM(10**8)
    hbm = HBM(10**8)
    start = Start(10**8)

    Hardware.intercon = {
        (HBM, SRAM): HardwareConnection(sram.clock_period, HBM_READ + SRAM_WRITE),
        (SRAM, HBM): HardwareConnection(DAC_ADC_DELAY, SRAM_READ + HBM_WRITE),
        # start nodes
        (start, CPU): HardwareConnection(0, 0),
        (start, PHU): HardwareConnection(0, 0),
        (start, SRAM): HardwareConnection(0, 0),
    }
    for hw_obj in hardware:
        if isinstance(hw_obj, CPU):
            Hardware.intercon.update({
                (SRAM, CPU): HardwareConnection(
                    sram.clock_period, SRAM_READ + LOCAL_WRITE + LOCAL_READ
                ),
                (CPU, SRAM): HardwareConnection(
                    sram.clock_period, LOCAL_WRITE + LOCAL_READ + SRAM_WRITE
                ),
            })
        if isinstance(hw_obj, PHU):
            Hardware.intercon.update({
                (SRAM, PHU): HardwareConnection(
                    sram.clock_period + DAC_ADC_DELAY, SRAM_READ + LOCAL_WRITE + LOCAL_READ + DAC_POWER
                ),
                (PHU, SRAM): HardwareConnection(
                    sram.clock_period + DAC_ADC_DELAY, ADC_POWER + LOCAL_WRITE + LOCAL_READ + SRAM_WRITE
                ),
            })



class HardwareConnection:
    def __init__(self, time_cost_per_transfer, energy_cost_per_number):
        """create functions for cost

        Args:
            time_cost_func (int): time cost in seconds per bit
            energy_cost_func (int): energy cost in joules per bit
        """
        self.time_cost_func = lambda n, b: time_cost_per_transfer
        self.energy_cost_func = lambda n, b: n * energy_cost_per_number
        self.var_to_func = {
            "time": self.time_cost_func,
            "energy": self.energy_cost_func,
        }

    def get_transfer_cost(
        self, weight_variable, num_transfer, bit_transfer
    ):
        return self.var_to_func[weight_variable](num_transfer, bit_transfer)


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
    algs = {}
    intercon = {}
    def __init__(self, clock_speed):
        self.clock_speed = clock_speed
        self.clock_period = 1 / clock_speed
        Hardware._universal_hardware_ID += 1
        self._initialize_algs()

    def __hash__(self):
        return hash(Hardware._universal_hardware_ID)

    def _initialize_algs(self):
        Hardware.algs.update(self.algs)


class PHU(Hardware):
    def __init__(self, clock_speed, num_cores, num_multiplex):
        self.num_numtiplex = num_multiplex
        self.num_cores = num_cores
        self.mac_energy = 0.04 * PICO_JOULE
        self.algs = {
            "task_para_matmul_phu": HardwareAlgorithm(
                "matmul",
                {self:
                (self._phu_matmul_task_para_cycles,
                self._phu_matmul_task_para_energy)
                }
            ),
            "task_para_dense_phu": HardwareAlgorithm(
                "dense",
                {self:
                (self._phu_matmul_task_para_cycles,
                self._phu_matmul_task_para_energy)
                }
            ),
            "task_para_pack_phu": HardwareAlgorithm(
                "pack",
                {self:
                (self._phu_matmul_task_para_cycles,
                self._phu_matmul_task_para_energy)
                }
            ),
            "dot_prod_phu": HardwareAlgorithm(
                "dot_prod",
                {self:
                (lambda i, o: i[0][-1],
                lambda i, o: i[0][-1] * self.mac_energy)
                }
            ),
        }
        super().__init__(clock_speed)

    def _phu_matmul_task_para_cycles(self, i, o):
        num_dot_products = ten_elm(o[0])
        length_dot_products = i[0][-1]
        phu_cycles = (
            math.ceil(math.ceil(num_dot_products / self.num_numtiplex) / self.num_cores)
            * length_dot_products
        )
        return phu_cycles

    def _phu_matmul_task_para_energy(self, i, o):
        num_dot_products = ten_elm(o[0])
        length_dot_products = i[0][-1]
        return num_dot_products * length_dot_products * self.mac_energy

class CPU(Hardware):
    def __init__(self, clock_speed, num_cores):
        self.num_cores = num_cores
        self.mac_energy = 0.1 * PICO_JOULE
        self.algs = {
            "add": HardwareAlgorithm("add", {self: (all_elm, standard_energy)} ),
            "subtract": HardwareAlgorithm("subtract", {self: (all_elm, standard_energy)} ),
            "multiply": HardwareAlgorithm("multiply", {self: (all_elm, standard_energy)} ),
            "divide": HardwareAlgorithm("divide", {self: (all_elm, standard_energy)} ),
            "sqrt": HardwareAlgorithm("sqrt", {self: (all_elm, standard_energy)} ),
            "rsqrt": HardwareAlgorithm("rsqrt", {self: (all_elm, standard_energy)} ),
            "relu": HardwareAlgorithm("relu", {self: (all_elm, standard_energy)} ),
            "tanh": HardwareAlgorithm("tanh", {self: (lambda i, o: ten_elm(o[0]) * 4, standard_energy) } ),

            "power": HardwareAlgorithm("power", {self: (all_elm, standard_energy) }),
            "transpose": HardwareAlgorithm("transpose", {self: (all_elm, standard_energy)} ),
            "nop": HardwareAlgorithm("nop", {self: (constnat(1), standard_energy)} ),
            "less": HardwareAlgorithm("less", {self: (constnat(1), standard_energy)} ),
            "take": HardwareAlgorithm("take", {self: (constnat(1), standard_energy)} ),
            "mean": HardwareAlgorithm(
                "mean", {self: (lambda i, o: (i[0][-1] + 1) * i[0][-2], standard_energy) }),
            "softmax": HardwareAlgorithm(
                "softmax", {self: (lambda i, o: ten_elm(o[0]) * 6, standard_energy) }
            ),
            "matmul": HardwareAlgorithm(
                "matmul", {self: (self._cpu_matmul_cycles, self._cpu_matmul_energy)}
            ),
            "dense": HardwareAlgorithm(
                "dense", {self: (self._cpu_matmul_cycles, self._cpu_matmul_energy)}
            ),
            "pack": HardwareAlgorithm(
                "pack", {self: (self._cpu_matmul_cycles, self._cpu_matmul_energy)}
            ),
            "where": HardwareAlgorithm("where", {self: (constnat(1), standard_energy) }),
            "erf": HardwareAlgorithm("erf",  {self: (constnat(1), standard_energy) })
        }
        super().__init__(clock_speed)

    def _cpu_matmul_cycles(self, i, o):
        num_dot_products = ten_elm(o[0])
        length_dot_products = i[0][-1]
        ops_per_mac = 2  # multiplicaiton and addition
        return num_dot_products * length_dot_products * ops_per_mac

    def _cpu_matmul_energy(self, i, o):
        num_dot_products = ten_elm(o[0])
        length_dot_products = i[0][-1]
        return num_dot_products * length_dot_products * self.mac_energy

class GPU(Hardware):
    def __init__(self, clock_speed):
        self.mac_energy = 0.1 * PICO_JOULE
        self.algs = {}
        super().__init__(clock_speed)


class HBM(Hardware):
    def __init__(self, clock_speed):
        self.algs = {
            "get_dram": HardwareAlgorithm("null", {self:( lambda i, o: 0 , lambda i, o: sum(ten_elm(a) for a in o) * DRAM_READ + sum(ten_elm(a) for a in o) * HBM_WRITE) }),
            # 0 if assuming model is preloaded to HMB
            # sum(ten_elm(a) for a in o) * BITS_PER_NUM / MEMORY_TRANSFER_WIDTH
        }
        super().__init__(clock_speed)


class SRAM(Hardware):
    def __init__(self, clock_speed):
        self.algs = {
            "split": HardwareAlgorithm("split", {self: (constnat(1), standard_energy)})
        }
        super().__init__(clock_speed)

class Start(Hardware):
    def __init__(self, clock_speed):
        self.algs = {
            "start": HardwareAlgorithm("start", {self: (constnat(1), standard_energy)}),
        }
        super().__init__(clock_speed)




# endregion
