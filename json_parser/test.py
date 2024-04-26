import math


def batch_vector(vector_size, num_batches):
    """Generator object that groups vectors into batches
    Args:
        vector_size (int): size of original vector
        batch_size (int): num hardware
    Yields:
        str: "[start_index: end index]"
    """
    temp = vector_size
    batch_size = math.ceil(vector_size/ num_batches)
    start = 0
    end = batch_size
    for i in range(num_batches):
        if temp < batch_size:
            end = start + temp
        yield [start, end]
        start += batch_size
        end += batch_size
        temp -= batch_size


batch_gen = batch_vector(784, 100)
print(list(batch_gen))
batch_gen = batch_vector(784, 100)
print(len(list(batch_gen)))
