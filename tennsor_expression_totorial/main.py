# Ryan Tomich
# Following Working with Operators Using Tensor Expression [https://tvm.apache.org/docs/tutorial/tensor_expr_get_started.html]
import tvm
import tvm.testing
from tvm import te #tensor Expression
import numpy as np


#################
#Vector Addition#
#################

tgt = tvm.target.Target(target="llvm", host="llvm") # Establishing Target device
    # device = llvm, to be exicuted on llvm. For llvm compatable devicces

n = te.var("n") #shape of the tensor
A = te.placeholder((n,), name="A") # Placeholder tensor size n
B = te.placeholder((n,), name="B") # Placeholder tensor size n
C = te.compute(A.shape, lambda i: A[i] + B[i], name="C") # Resulting tensor
    # lambda function is doing the oporation. For vector addition, we are adding each individual elemtn
    #   in tensor A to tensor B
    # no calculation has been done yet becasue we havent given it a tensor. This just establishes the inferstraucture.

### Schedulgin ###
    # Compute C in row-major order. In order of i
    # creats a schedule "s"
s = te.create_schedule(C.op)

### Building ###
    # making a python-runable function
fadd = tvm.build(s, [A, B, C], tgt, name="myadd")
    # tvm. build(schedule, function signature,target language, name)

### Running ###
dev = tvm.device(tgt.kind.name, 0)

n = 1024
a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)
fadd(a, b, c)
tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())
    # PASSES - meaning the function we made using TVM TE produced the same result as numpy's
    #   add vector function.

### Time Comparison###
import timeit

np_repeat = 100
np_running_time = timeit.timeit(
    setup="import numpy\n"
    "n = 32768\n"
    'dtype = "float32"\n'
    "a = numpy.random.rand(n, 1).astype(dtype)\n"
    "b = numpy.random.rand(n, 1).astype(dtype)\n",
    stmt="answer = a + b",
    number=np_repeat,
)
print("Numpy running time: %f" % (np_running_time / np_repeat))


def evaluate_addition(func, target, optimization, log):
    dev = tvm.device(target.kind.name, 0)
    n = 32768
    a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
    b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
    c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)

    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    mean_time = evaluator(a, b, c).mean
    print("%s: %f" % (optimization, mean_time))

    log.append((optimization, mean_time))


log = [("numpy", np_running_time / np_repeat)]
evaluate_addition(fadd, tgt, "naive", log=log)
