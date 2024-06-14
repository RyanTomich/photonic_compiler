import numpy as np

def ten_elm(tensor_shape):
    ans = 1
    for dimention in tensor_shape:
        ans *= dimention
    return ans

def elm_const(matrix, const = 1):
    return ten_elm(matrix) * const

opp_time_func = {
    'add':      lambda i, o: (elm_const(o[0]), np.inf ),
    'subtract': lambda i, o: ( elm_const(o[0]) , np.inf),
    'multiply': lambda i, o: ( elm_const(o[0]) , np.inf),
    'divide':   lambda i, o: (  elm_const(o[0]) , np.inf),
    'sqrt':     lambda i, o: ( elm_const(o[0]) , np.inf),
    'rsqrt':    lambda i, o: ( elm_const(o[0]) , np.inf),
    'tanh':     lambda i, o: ( elm_const(o[0],4), np.inf), #e^x definition
    'power':    lambda i, o: ( elm_const(o[0]) , np.inf),
    'nop':      lambda i, o: ( 0, np.inf), #reshape(Non-Computational)
    'less':     lambda i, o: ( 1, np.inf),
    'where':    lambda i, o: ( 1, np.inf),
    'take':     lambda i, o: ( 1, np.inf),
    'split':    lambda i, o: ( 3, np.inf), # for each split
    'transpose':lambda i, o: ( elm_const(o[0]) , np.inf), # unsture
    'mean':     lambda i, o: ( (i[0][-1]+1)* i[0][-2], np.inf),
    'softmax':  lambda i, o: ( 6*i[0][-1]*i[0][-2], np.inf),
    'matmul':   lambda i, o: ( ten_elm(i[0])*i[1][-2]*2, ten_elm(i[0])*i[1][-2]*2),
    'dense':    lambda i, o: ( ten_elm(i[0])*i[1][-2]*2, ten_elm(i[0])*i[1][-2]*2),
    'pack':     lambda i, o: ( ten_elm(i[0])*i[1][-2]*2, ten_elm(i[0])*i[1][-2]*2), # another form of dense
}
