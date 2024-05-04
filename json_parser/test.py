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
    remainder = vector_size % num_batches
    print(f'{batch_size=}')
    print(f'{remainder=}')
    start = 0
    end = batch_size
    for i in range(num_batches):
        print(batch_size)
        if remainder == 0:
            batch_size = batch_size-1
            end = end-1
        # if temp < batch_size:
        #     end = start + temp
        yield [start, end]
        start += batch_size
        end += batch_size
        temp -= batch_size
        remainder -= 1


batch_gen = batch_vector(784,10)
ans = list(batch_gen)
print(len(ans))
print(ans)
# batch_gen = batch_vector(784, 100)
# print(len(list(batch_gen)))
