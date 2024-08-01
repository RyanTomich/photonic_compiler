import torch
import os

from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn as nn
import torch.nn.functional as F

import psutil

def print_inference_data(model, inputs, detailed=False, trace=False):
    """
    https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
        chrome://tracing/
    """



    print("CPU")
    with profile(
        activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True
    ) as prof:
        with record_function("model_inference"):
            model(inputs)

    if detailed:
        print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total"))
    else:
        print(prof.key_averages().table(sort_by="cpu_time_total"))


    if trace:
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            model(inputs)

        prof.export_chrome_trace("trace.json")


os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)

print(torch.get_num_threads())
print(f'{psutil.cpu_freq()} MHz')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LeNet(nn.Module):
    def __init__(self, mask=False):
        super(LeNet, self).__init__()
        linear = MaskedLinear if mask else nn.Linear
        self.fc1 = linear(784, 300, bias=False)
        self.fc2 = linear(300, 100, bias=False)
        self.fc3 = linear(100, 10, bias=False)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x



model = LeNet().to(device=device)

print(model)
input_shape = [28, 28]
input_data = torch.randn(input_shape)


print_inference_data(model, input_data, detailed=True)


# def print_cpu_info():
#     print(f"Number of physical cores: {psutil.cpu_count(logical=False)}")
#     print(f"Number of logical cores: {psutil.cpu_count(logical=True)}")


# print_cpu_info()
