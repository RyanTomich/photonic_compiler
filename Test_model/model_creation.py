import tensorflow as tf
import numpy as np
import keras
from keras import layers

def get_model():
    '''
    in: (1,4)
    ount: (1,1)
    define sequential model with three layers
    '''
    model = keras.Sequential(name="sequential_modle")
    model.add(keras.Input(shape=(4,)))
    model.add(layers.Dense(2, name="layer1"))
    model.add(layers.Dense(3, name="layer2"))
    model.add(layers.Dense(1, name="layer3"))
    model.compile(optimizer=keras.optimizers.Adam(), loss="mean_squared_error")
    return model

model = get_model()

# Training
test_input = np.random.random((1,4))
test_target = np.random.random((1,1))
print(test_input, test_target)
model.fit(test_input, test_target)

# save the model
model.save("test_model", save_format='tf')


# Pull rile
reconstructed_model = keras.models.load_model("test_model")
print(model.predict(test_input))
print(reconstructed_model.predict(test_input))

input_data = np.array([[1, 2, 3, 4]])
output = model.predict(input_data)

print(f"{input_data}")
print(f"{output}")
