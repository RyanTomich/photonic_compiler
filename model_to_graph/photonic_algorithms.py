# def create_dot_product(m1, m2):
#     for row in range(m1[-2]):
#         for col in range(m2[-1]):
#             yield f'm1[{row}, :] * m2[: ,{col}]  -  {m1[-2]}'

# a = 2
# b = 2
# c = 3
# d = 4
# m1 = (a, b, c)
# m2  = (a, c, d)

# dots = list(create_dot_product(m1, m2))
# print(len(dots))
# print(a*b*d)
# for i in dots:
#     print (i)



def matmul(m1, m2, preamble):
    for row in range(m1[-2]):
        for col in range(m2[-1]):

            yield f'm1{preamble + [row, ':']} * m2{preamble + [':', col]}  -  {m1[-1]}'


def nd_tensor_product(m1, m2, preamble = []):
    if len(m1) == 2:
       yield from matmul(m1, m2, preamble)
    else:
        for dimention in range(m1[0]):
            preamble.append(dimention)
            yield from nd_tensor_product(m1[1:], m2[1:], preamble=preamble)
            preamble.pop()

m1 = (2, 2, 3)
m2 = (2, 3, 3)

for i in list(nd_tensor_product(m1, m2)):
    print (i)
