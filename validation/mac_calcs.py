import os
import psutil
import numpy as np
import time


# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"

# cpu_freq = psutil.cpu_freq()
# print(f"CPU Frequency: {cpu_freq.current} MHz")

# times = []
# for i in range (100):
#     a = random_float32 = np.float32(np.random.rand())
#     b = random_float32 = np.float32(np.random.rand())
#     t1 = time.time()
#     a * b
#     t2 = time.time()
#     times.append(t2-t1)

# print(sum(times)/len(times))


def per_mac(num_dot_prod, len_dot_prod, dur):
    time_per_dot = dur / num_dot_prod
    # print(f"{time_per_dot=}")
    time_per_mac = time_per_dot / len_dot_prod
    print(f"{time_per_mac=}")


# dense
num_dot_prod = 9216
len_dot_prod = 768
dur = 506.27 * (10**-6)
per_mac(num_dot_prod, len_dot_prod, dur)

# dense_1
num_dot_prod = 3072
len_dot_prod = 768
dur = 158.909 * (10**-6)
per_mac(num_dot_prod, len_dot_prod, dur)

# dense_2
num_dot_prod = 12288
len_dot_prod = 768
dur = 729.5084375 * (10**-6)
per_mac(num_dot_prod, len_dot_prod, dur)

# dense_3
num_dot_prod = 3072
len_dot_prod = 3072
dur = 715.441 * (10**-6)
per_mac(num_dot_prod, len_dot_prod, dur)


# dense_4
num_dot_prod = 201028
len_dot_prod = 768
dur = 36106.88938 * (10**-6)
per_mac(num_dot_prod, len_dot_prod, dur)

# pack
num_dot_prod = 4608
len_dot_prod = 768
dur = 199.438 * (10**-6)
per_mac(num_dot_prod, len_dot_prod, dur)

# pack 1
num_dot_prod = 18432
len_dot_prod = 768
dur = 890.188 * (10**-6)
per_mac(num_dot_prod, len_dot_prod, dur)

# pack 2
num_dot_prod = 4608
len_dot_prod = 3072
dur = 877.281 * (10**-6)
per_mac(num_dot_prod, len_dot_prod, dur)

# pack 3
num_dot_prod = 183132
len_dot_prod = 768
dur = 36093.823 * (10**-6)
per_mac(num_dot_prod, len_dot_prod, dur)
