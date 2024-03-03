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
    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square conv kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5x5 image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = LeNet().to(device=device)
torch.save(model, 'LeNet.pt')
print(model)
model = model.eval()

input_shape = [1, 1, 32, 32]
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
