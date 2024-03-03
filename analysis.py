import onnxruntime
import tensorflow as tf
from tvm.driver import tvmc
import torch
from torchsummary import summary

### ONNX
# # load the onnx model
# onnx_model_path = 'bvlcalexnet-12.onnx'
# session = onnxruntime.InferenceSession(onnx_model_path)

# # get input layer name and size
# input_name = session.get_inputs()[0].name
# input_shape = session.get_inputs()[0].shape

# print(f"{input_name}, {input_shape}")


### Tensorflow
# model = tf.keras.models.load_model("Test_model/test_model")

# input_layer = model.layers[0]
# input_name = input_layer.name
# input_shape = model.layers[0].input_shape

# print(f"{input_name}: {input_shape}")

# ### Pytorch
model = torch.load("Pytorch/LeNet.pt")

summary(model, (1, 13, 13))



### TVM Things
# model = tvmc.load('Pytorch/LeNet.pt')
# model.summary()
# package = tvmc.compile(model, target = "llvm")
