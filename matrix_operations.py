import math


def zero_matrix(n):
    A = []
    for i in range(n):
        A.append([0.0] * n)
    return A


def identity_matrix(n):
    E = zero_matrix(n)
    for i in range(n):
        E[i][i] = 1.0
    return E


def copy_matrix(A):
    return [row.copy() for row in A]


def transpose(A):
    n = len(A)
    result = zero_matrix(n)
    for i in range(n):
        for j in range(n):
            result[j][i] = A[i][j]
    return result


def matrix_product(A, B):
    n = len(A)
    result = zero_matrix(n)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                result[i][j] += A[i][k] * B[k][j]
    return result


def matrix_vector_product(A, x):
    n = len(A)
    result = [0.0] * n
    for i in range(n):
        for j in range(n):
            result[i] += A[i][j] * x[j]
    return result


def vector_norm(x):
    return max(abs(value) for value in x)


def matrix_norm(A):
    return max(sum(abs(value) for value in row) for row in A)


def vector_difference(x, y):
    return max(abs(x[i] - y[i]) for i in range(len(x)))


def matrix_difference(A, B):
    error = 0.0
    for i in range(len(A)):
        for j in range(len(A)):
            error = max(error, abs(A[i][j] - B[i][j]))
    return error


def off_diagonal_norm(A):
    t = 0.0
    for i in range(len(A)):
        for j in range(i + 1, len(A)):
            t = math.hypot(t, A[i][j])
    return t
