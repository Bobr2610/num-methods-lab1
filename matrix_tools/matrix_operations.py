from math import hypot


def zero_matrix(
        n: int
) -> list[list[float]]:
    A = []
    for i in range(n):
        A.append([0.0] * n)
    return A


def identity_matrix(
        n: int
) -> list[list[float]]:
    E = zero_matrix(n)
    for i in range(n):
        E[i][i] = 1.0
    return E


def transpose(
        A: list[list[float]]
) -> list[list[float]]:
    n = len(A)
    result = zero_matrix(n)
    for i in range(n):
        for j in range(n):
            result[j][i] = A[i][j]
    return result


def matrix_product(
        A: list[list[float]],
        B: list[list[float]]
) -> list[list[float]]:
    n = len(A)
    result = zero_matrix(n)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                result[i][j] += A[i][k] * B[k][j]
    return result


def matrix_vector_product(
        A: list[list[float]],
        x: list[float]
) -> list[float]:
    n = len(A)
    result = [0.0] * n
    for i in range(n):
        for j in range(n):
            result[i] += A[i][j] * x[j]
    return result


def vector_norm(
        x: list[float]
) -> float:
    return max(abs(value) for value in x)


def matrix_norm(
        A: list[list[float]]
) -> float:
    return max(sum(abs(value) for value in row) for row in A)


def vector_difference(
        x: list[float],
        y: list[float]
) -> float:
    return max(abs(x[i] - y[i]) for i in range(len(x)))


def matrix_difference(
        A: list[list[float]],
        B: list[list[float]]
) -> float:
    error = 0.0
    for i in range(len(A)):
        for j in range(len(A)):
            error = max(error, abs(A[i][j] - B[i][j]))
    return error


def matrices_equal(
        A: list[list[float]],
        B: list[list[float]],
        eps: float = 1e-9
) -> bool:
    if len(A) != len(B):
        return False
    for i in range(len(A)):
        if len(A[i]) != len(B[i]):
            return False
        for j in range(len(A[i])):
            if abs(A[i][j] - B[i][j]) > eps:
                return False
    return True


def off_diagonal_norm(
        A: list[list[float]]
) -> float:
    t = 0.0
    for i in range(len(A)):
        for j in range(i + 1, len(A)):
            t = hypot(t, A[i][j])
    return t
