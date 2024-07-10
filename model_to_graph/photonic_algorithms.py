import operator_calcs as oc

# Create Dot Products
def matmul(m1, m2, preamble):
    """createss dot products from 2 matricies

    Args:
        m1 (tuple): shape of m1
        m2 (tuple): shape of m2
        preamble (list): list of indexing path to current matrix.

    Yields:
        srt: dot product with indexing
    """
    for row in range(m1[-2]):
        for col in range(m2[-2]):
            yield (preamble + (row, ':'), preamble + (col, ':'), m1[-1]) # (v1, v2, size)
            # yield f'm1{preamble + [row, ':']} * m2{preamble + [col, ':']}  -  {m1[-1]}'

def nd_tensor_product(m1, m2, preamble = ()):
    """Breaks tensor down to the level of matrix multiplication

    Args:
        m1 (tuple): shape of m1
        m2 (tuple): shape of m2
        preamble (list, optional): list of indexing path to current matrix. Defaults to [].

    Yields:
        srt: dot product with indexing
    """
    if len(m1) == 2:
       yield from matmul(m1, m2, preamble)
    else:
        for dimention in range(m1[0]):
            preamble = preamble + (dimention,)
            yield from nd_tensor_product(m1[1:], m2[1:], preamble=preamble)
            preamble = preamble[:-1]

def multiplex_groups(common_operand, unique_operands):
    full = int(len(unique_operands)/oc.PHU_MULTIPLEX)
    start = 0
    end = oc.PHU_MULTIPLEX
    for i in range(full):
        yield (common_operand, unique_operands[start:end])
        start = end
        end += oc.PHU_MULTIPLEX

    assert end >= len(unique_operands)
    if len(unique_operands)%oc.PHU_MULTIPLEX:
        assert end > len(unique_operands)
        yield (common_operand, unique_operands[start:])




def task_para_node_gen(common_operand, unique_operands):
    # unique = {}
    for multiplex in multiplex_groups(common_operand, unique_operands):
        # dot = f'{multiplex[0]} - {len(multiplex[1])}'
        # unique.setdefault(dot, 0)
        # unique[dot] += 1


def dynamic_para_node_gen(common_operand, unique_operands):
    pass


node_expansion = {
    'task_para_matmul_phu': task_para_node_gen,
    'dynamic_para_matmul_phu': dynamic_para_node_gen,
    'task_para_dense_phu': task_para_node_gen,
    'dynamic_para_dense_phu': dynamic_para_node_gen,
    'task_para_pack_phu': task_para_node_gen,
    'dynamic_para_pack_phu': dynamic_para_node_gen,
}
