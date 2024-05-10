import pytest
import random
from parser import looper, batch_vector


@pytest.mark.parametrize("input_value, expected_output", [
    (3, [1, 2, 3, 1, 2]),
    (10, [1, 2, 3, 4, 5]),
    (1, [1, 1, 1, 1, 1]),
])
def test_looper(input_value, expected_output):
    looper_gen = looper(input_value)
    ans = [next(looper_gen) for _ in range(5)]
    assert ans == expected_output

def test_looper_with_zero_input():
    with pytest.raises(AssertionError):
        looper_gen = looper(0)
        next(looper_gen)

def test_batch_vector():

    batch_vec_1_ans = [
        [0,10],
        [10,20],
        [20,30],
        [30,40],
        [40,50],
        [50,60],
        [60,70],
        [70,80],
        [80,90],
        [90,100]]
    batch_vec_2_ans = [
        [0, 12],
        [12, 24],
        [24, 35],
        [35, 46],
        [46, 57],
        [57, 68],
        [68, 79],
        [79, 90],
        [90, 101],
        [101, 112]]
    batch_vec_3_ans = [
        [0,1],
        [1,2],
        [2,3],
        [3,4],
        [4,5]]

    vector_size = 100
    num_batches = 10
    assert list(batch_vector(vector_size, num_batches)) == batch_vec_1_ans

    vector_size = 112
    num_batches = 10
    assert list(batch_vector(vector_size, num_batches)) == batch_vec_2_ans

    vector_size = 5
    num_batches = 5
    assert list(batch_vector(vector_size, num_batches)) == batch_vec_3_ans

def test_batch_vector_random():
    for _ in range(10):
        vector_size = random.randint(1, 100)
        num_batches = random.randint(1, vector_size)

        print(vector_size, num_batches)

        batch = list(batch_vector(vector_size, num_batches))
        assert len(batch) == num_batches
        assert batch[-1][1] == vector_size
        for pair in batch:
            assert pair[0] < pair[1]
