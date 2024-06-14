import numpy as np

# Constants
photonic_opps = ['dense', 'matmul']
CPU_CLOCK_SPEED = 10**8 #.1Ghz
PHU_CLOCK_SPEED = 10**10 # 10 Ghz
cycle_to_time_funcs = {'CPU': lambda x: x / CPU_CLOCK_SPEED,
                       'PHU': lambda x: x / PHU_CLOCK_SPEED}

def ten_elm(tensor_shape):
    ans = 1
    for dimention in tensor_shape:
        ans *= dimention
    return ans

def elm_const(matrix, const = 1):
    return ten_elm(matrix) * const



all_elm = lambda i, o: {"CPU": ten_elm(o[0])}
constnat = lambda c: lambda i, o:  {"CPU": c}

# 'mean':     lambda i, o: ( (i[0][-1]+1)* i[0][-2], np.inf),
# 'softmax':  lambda i, o: ( 6*i[0][-1]*i[0][-2], np.inf),
# 'matmul':   lambda i, o: ( ten_elm(i[0])*i[1][-2]*2, ten_elm(i[0])*i[1][-2]*2),
# 'dense':    lambda i, o: ( ten_elm(i[0])*i[1][-2]*2, ten_elm(i[0])*i[1][-2]*2),
# 'pack':     lambda i, o: ( ten_elm(i[0])*i[1][-2]*2, ten_elm(i[0])*i[1][-2]*2), # another form of dense

# Overlay optimization in formulas here
run_cpu_cycles = {
    'add':      all_elm,
    'subtract': all_elm,
    'multiply': all_elm,
    'divide':   all_elm,
    'sqrt':     all_elm,
    'rsqrt':    all_elm,
    'tanh':     lambda i, o: {"CPU": ten_elm(o[0])*4}, #e^x definition
    'power':    all_elm,
    'transpose':all_elm,
    'nop':      constnat(0),
    'less':     constnat(1),
    'take':     constnat(1),
    'split':    constnat(3),
    'mean':     constnat(1),
    'softmax':  constnat(1),
    'matmul':   constnat(1),
    'dense':    constnat(1),
    'where':    constnat(1),
    'pack':     constnat(1),
}

run_phu_cycles = {
    'matmul': lambda i,o: {"CPU":2, "PHU": 1},
    'dense': lambda i,o: {"CPU":2, "PHU": 1},
    'pack': lambda i,o: {"CPU":2, "PHU": 1}
}

cycle_funcions = {"run_cpu": run_cpu_cycles, "run_phu": run_phu_cycles}
hardware_format = {"run_cpu": {"CPU": None} ,"run_phu": {"CPU": None, "PHU": None} }


def opp_time_func(opp, input_shapes, output_shapes, run_hardware):
    '''
    get the total clock cycles for each hardware choice
    opp(srt)
    input_shapes(list of list)
    output_shapes(list of list)
    hardware choice[str]

    matriceis representing cost of hardware choice
    ["CPU", #, # ...]

    [
        ["CPU", #, # ...]
        ["PHU", #, # ...]
    ]
    '''
    opp_cycle_dict = cycle_funcions[run_hardware]
    format = hardware_format[run_hardware]

    if opp in opp_cycle_dict:
        cycles = opp_cycle_dict[opp](input_shapes, output_shapes)
        for hardware, cost in cycles.items():
            format[hardware] = cost
        time_total = 0
        for hardware, cycles in format.items():
            time_total += cycle_to_time_funcs[hardware](cycles)
        return time_total

    return np.inf
