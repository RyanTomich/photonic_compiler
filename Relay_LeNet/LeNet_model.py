# to explore
# Relay: A New IR for Machine Learning Frameworks. (Roesch et al., 2018), https://doi.org/10.1145/3211346.3211348
# Gradient-Based Learning Applied to Document Recognition. (YANN LECUN et al., 1998) https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=726791


@relay_model
def lenet(x: Tensor[Float, (1, 28, 28)]) -> Tensor[Float, 10]:
    conv1 = relay.conv2d(x, num_filter=20, ksize=[1, 5, 5, 1], no_bias=False)
    tanh1 = relay.tanh(conv1)
    pool1 = relay.max_pool(tanh1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])
    conv2 = relay.conv2d(pool1, num_filter=50, ksize=[1, 5, 5, 1], no_bias=False)
    tanh2 = relay.tanh(conv2)
    pool2 = relay.max_pool(tanh2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])
    flatten = relay.flatten_layer(pool2)
    fc1 = relay.linear(flatten, num_hidden=500)
    tanh3 = relay.tanh(fc1)
    return relay.linear(tanh3, num_hidden=10)
@relay
def loss(x: Tensor[Float, (1, 28, 28)], y: Tensor[Float, 10]) -> Float:
    return relay.softmax_cross_entropy(lenet(x), y)
@relay
def train_lenet(training_data: Tensor[Float, (60000, 1, 28, 28)]) -> Model:
    model = relay.create_model(lenet)
    for x, y in data:
        model_grad = relay.grad(model, loss, (x, y))
        relay.update_model_params(model, model_grad)
    return relay.export_model(model)

training_data, test_data = relay.datasets.mnist()
model = train_lenet(training_data)
print(relay.argmax(model(test_data[0])))
