import numpy as np
import math


def absmax_quantization(m1, int_type):
    data_max = np.iinfo(int_type).max
    norm_range = math.floor((data_max)**(1/2)/ max(m1.shape))

    quantification_constant = norm_range/np.max(abs(m1))
    new_type = (m1 * quantification_constant).astype(int_type)
    assert np.all((new_type >= -1*norm_range) & (new_type <= norm_range))
    return new_type, quantification_constant

def zeropoint_quantization(x, int_type):
    data_range = np.iinfo(int_type).max- np.iinfo(int_type).min
    q_c = data_range / (np.max(x)- np. min(x))
    z_p = -1* np.round(q_c * np.min(x)) + np.iinfo(int_type).min
    type_x = np.round(q_c * x + z_p).astype(int_type)
    return type_x, q_c, z_p

def absmax_quantization_dot_product(row, col, qualtization_type):
    split_row = np.array_split(row, round(len(row)/2))
    split_col = np.array_split(col, round(len(row)/2))

    float_dot = 0
    for split_idx in range(len(split_row)):
        int_row, row_const = absmax_quantization(split_row[split_idx], qualtization_type)
        int_col, col_const = absmax_quantization(split_col[split_idx], qualtization_type)
        int_dot = np.dot(int_row, int_col)
        float_dot += int_dot.astype(np.float32)* 1/(row_const * col_const)
    return float_dot


def vector_quant_matmul(m1, m2, quantization_method, qualtization_type):
    if quantization_method == 'matrix':
        int_m1, m1_const= absmax_quantization(m1, qualtization_type)
        int_m2, m2_const = absmax_quantization(m2, qualtization_type)
        ans = np.matmul(int_m1, int_m2)
        return ans.astype(np.float32)* 1/(m1_const * m2_const)

    matrix = np.empty((m1.shape[0], m2.shape[-1]))
    for row_num in range(len(m1)):
        for col_num in range(m2.shape[-1]):
            row = m1[row_num]
            col = m2[:, col_num]
            if quantization_method == 'absmax_split':
                float_dot = absmax_quantization_dot_product(row, col, qualtization_type)
            elif quantization_method == 'absmax':
                int_row, row_const = absmax_quantization(row, qualtization_type)
                int_col, col_const = absmax_quantization(col, qualtization_type)
                int_dot = np.dot(int_row, int_col)
                float_dot = int_dot.astype(np.float32)* 1/(row_const * col_const)

            elif quantization_method == 'zeropoint':
                int_row, row_const, row_zero = zeropoint_quantization(row, qualtization_type)
                int_col, col_const, col_zero = zeropoint_quantization(col, qualtization_type)
                int_dot = np.dot(int_row - row_zero, int_col- col_zero) #TODO Still multipluing in float 64
                float_dot = (int_dot /(row_const * col_const)).astype(np.float32)

            if np.isnan(float_dot):
                float_dot = 0

            matrix[row_num][col_num] = float_dot
    return matrix

def sepperate_outliers(m1, m2, quantization_method, qualtization_type):
    print(f"start: {m1.shape}")
    col_sums = []
    for i in range(m1.shape[-1]):
        col_sums.append(np.abs(sum(m1[:, i])))
    col_sum_mean = np.mean(col_sums)

    m1_outliers = np.empty((m1.shape[0], 0))
    m2_outliers = np.empty((0 ,m2.shape[-1]))
    m1_inliers = np.empty((m1.shape[0], 0))
    m2_inliers = np.empty((0 ,m2.shape[-1]))

    for index, item in enumerate(col_sums):
        if item > col_sum_mean*10:
            m1_outliers = np.hstack((m1_outliers, m1[:, index][:, np.newaxis]))
            m2_outliers = np.vstack((m2_outliers, m2[index]))
        else:
            m1_inliers = np.hstack((m1_inliers, m1[:, index][:, np.newaxis]))
            m2_inliers = np.vstack((m2_inliers, m2[index]))

    outliers = np.matmul(m1_outliers, m2_outliers)
    inliers = vector_quant_matmul(m1_inliers, m2_inliers, quantization_method, qualtization_type)

    return outliers + inliers


# def nd_tensor_product(m1, m2, quantization_method = 'absmax'):
#     ans_shape = m1.shape[:-2] + (m1.shape[-2], m2.shape[-1])
#     ans = np.empty(ans_shape)
#     if len(ans_shape) < 3:
#         return vector_quant_matmul(m1, m2, quantization_method)
#     else:
#         for i in range(ans_shape[0]):
#             ans[i] = nd_tensor_product(m1[i], m2[i], quantization_method=quantization_method)
#     return ans


# iterations = 1000
# qualtization_type = np.int8

# absmax_diff = 0
# absmax_split__diff = 0
# mtx_absmax_diff = 0

# zeropoint_smaller = 0
# for _ in range(iterations):
#     a = np.random.uniform(-20, 20, size=(2, 6, 9)).astype(np.float32)
#     b = np.random.uniform(-20, 20, size=(2, 9, 6)).astype(np.float32)

#     actual = a @ b

#     mtx_absmax_quantized = nd_tensor_product(a,b, quantization_method = 'matrix')
#     mtx_absmax_loss = np.abs(np.mean(mtx_absmax_quantized - actual))
#     mtx_absmax_diff += mtx_absmax_loss

#     absmax_quantized = nd_tensor_product(a,b, quantization_method = 'absmax')
#     absmax_loss = np.abs(np.mean(absmax_quantized - actual))
#     absmax_diff += absmax_loss

#     absmax_split_quantized = nd_tensor_product(a,b, quantization_method = 'absmax_split')
#     absmax_split_loss = np.abs(np.mean(absmax_split_quantized - actual))
#     absmax_split__diff += absmax_split_loss

# print(f'full matrix absmax loss {mtx_absmax_diff/iterations}')
# print(f'Vectorize absmax loss {absmax_diff/iterations}')
# print(f'Split vectorize absmax_split loss {absmax_split__diff/iterations}')
