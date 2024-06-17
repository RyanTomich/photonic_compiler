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
    'mean':     lambda i, o: {"CPU":(i[0][-1]+1)* i[0][-2]},
    'softmax':  lambda i, o: {"CPU": ten_elm(o[0])*6},
    'matmul':   lambda i, o: {"CPU": ten_elm(i[0])*i[1][-2]*2},
    'dense':    lambda i, o: {"CPU": ten_elm(i[0])*i[1][-2]*2},
    'pack':     lambda i, o: {"CPU": ten_elm(i[0])*i[1][-2]*2},
    'where':    constnat(1),
}

run_phu_cycles = {
    'matmul':  lambda i, o: {"PHU": ten_elm(i[0])*i[1][-2]},
    'dense':  lambda i, o: {"PHU": ten_elm(i[0])*i[1][-2]},
    'dense':  lambda i, o: {"PHU": ten_elm(i[0])*i[1][-2]},
    'pack':  lambda i, o: {"PHU": ten_elm(i[0])*i[1][-2]},
}

cycle_funcions = {"run_cpu": run_cpu_cycles, "run_phu": run_phu_cycles}


def opp_time_func(opp, input_shapes, output_shapes, hardware_config):
    '''
    get the total clock cycles for each hardware choice
    opp(srt)
    input_shapes(list of list)
    output_shapes(list of list)
    run_hardware[str]: which hardware to run computation.

    matriceis representing cost of hardware choice
    ["CPU", #, # ...]

    [
        ["CPU", #, # ...]
        ["PHU", #, # ...]
    ]
    '''
    opp_cycle_dict = cycle_funcions[hardware_config]

    if opp in opp_cycle_dict:
        cycles_dict = opp_cycle_dict[opp](input_shapes, output_shapes)
        time_total = 0
        for hardware, cycles in cycles_dict.items():
            time_total += cycle_to_time_funcs[hardware](cycles)
        return time_total

    return np.inf
