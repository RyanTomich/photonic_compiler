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



def run_cpu():
    print('1')
def run_phu():
    print('2')
def run_gpu():
    print('3')
def func():
    print('placeholder')


hw_intercon ={'CPU_CPU': lambda x: x*1,
              'CPU_PHU': lambda x: x*1,
              'CPU_GPU': lambda x: x*1,
              'PHU_PHU': lambda x: x*1,
              'PHU_GPU': lambda x: x*1,
              'GPU_GPU': lambda x: x*1,
}


hardware_algs = {
    'matmul_cpu': ('matmul', 'CPU', run_cpu, lambda x: x),
    'matmul_phu': ('matmul', 'PPU', run_phu, lambda x: x),
    'matmul_gpu': ('matmul', 'GPU', run_gpu, lambda x: x),
    'add' : ('add', 'CPU' , func, lambda x: 1),
    'subtract' : ('subtract', 'CPU' , func, lambda x: 1),
    'multiply' : ('multiply', 'CPU' , func, lambda x: 1),
    'divide' : ('divide', 'CPU' , func, lambda x: 1),
    'sqrt' : ('sqrt', 'CPU' , func, lambda x: 1),
    'rsqrt' : ('rsqrt', 'CPU' , func, lambda x: 1),
    'tanh' : ('tanh', 'CPU' , func, lambda x: 1),
    'power' : ('power', 'CPU' , func, lambda x: 1),
    'transpose' : ('transpose', 'CPU' , func, lambda x: 1),
    'nop' : ('nop', 'CPU' , func, lambda x: 1),
    'less' : ('less', 'CPU' , func, lambda x: 1),
    'take' : ('take', 'CPU' , func, lambda x: 1),
    'split' : ('split', 'CPU' , func, lambda x: 1),
    'mean' : ('mean', 'CPU' , func, lambda x: 1),
    'softmax' : ('softmax', 'CPU' , func, lambda x: 1),
    'matmul' : ('matmul', 'CPU' , func, lambda x: 1),
    'dense' : ('dense', 'CPU' , func, lambda x: 1),
    'pack' : ('pack', 'CPU' , func, lambda x: 1),
    'where' : ('where', 'CPU' , func, lambda x: 1),
}
