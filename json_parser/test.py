import math

def batch_vector(vector_size, batch_size, num_registers):
    temp = vector_size

    start = 0
    end = batch_size
    for i in range(num_registers):
        if temp < batch_size:
            end = start + temp
        yield f"[{start}:{end}]"
        start += batch_size
        end += batch_size
        temp -= batch_size

vector = [1,728]
max_array_len = 100
num_registers = math.ceil(vector[1] / max_array_len)

batch_gen = batch_vector(vector[1], max_array_len, num_registers)
for register in range(num_registers):
    print(f"   Register {register}: {next(batch_gen)}\n ")
