import torch.nn as nn
import torch.nn.functional as F
import torchvision

# pytorch
import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from torchsummary import summary
import torchvision

#tvm
import tvm
from tvm import relay
import numpy as np

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
torch.save(model, 'LeNet.pt')
print(model)
model = model.eval()

input_shape = [1, 1, 28, 28]
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(model, input_data).eval()


input_name = "conv1"
shape_list = [(input_name, input_shape)]
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

target = tvm.target.Target("llvm", host="llvm")
dev = tvm.cpu(0)
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params) #module


file_path = "LeNet_module.so"
lib.export_library(file_path)

# Save the graph JSON to a file
graph_json_path = "LeNet_graph.json"
with open(graph_json_path, "w") as f:
    f.write(lib.get_graph_json())

# Save the parameters to a file
param_dict = lib.get_params()  # No need to convert to dictionary
param_bytes_path = "LeNet_params.params"
tvm.relay.save_param_dict(param_dict)
