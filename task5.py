import cmath
from copy import deepcopy
from math import hypot, isfinite

import numpy as np

from matrix_tools import (
    format_number, identity_matrix, input_path, matrix_product,
    matrices_equal, print_matrix, print_vector, read_eps,
    read_matrix, read_size, read_tokens, require_end, require_token,
    transpose, use_utf8_output,
)


def read_problem(
        path: str
) -> tuple[list[list[float]], float, list[list[float]]]:
    tokens = read_tokens(path)
    n = read_size(tokens[0])
    A, start = read_matrix(tokens, n, 1)
    eps = read_eps(require_token(tokens, start))
    n = read_size(require_token(tokens, start + 1))
    example, end = read_matrix(tokens, n, start + 2)
    require_end(tokens, end)
    return A, eps, example


def householder_matrix(
        b: list[float]
) -> list[list[float]]:
    n = len(b)
    H = identity_matrix(n)
    length = hypot(*b)
    if length == 0:
        return H

    sign = 1.0 if b[0] >= 0 else -1.0
    v = [value / length for value in b]
    v[0] += sign

    denom = sum(value * value for value in v)
    for i in range(n):
        for j in range(n):
            H[i][j] -= 2 * v[i] * v[j] / denom
    return H


def qr_decomposition(
        A: list[list[float]]
) -> tuple[list[list[float]], list[list[float]]]:
    n = len(A)
    Q = identity_matrix(n)
    R = deepcopy(A)
    for k in range(n - 1):
        b = [R[i][k] for i in range(k, n)]
        block = householder_matrix(b)
        H = identity_matrix(n)
        for i in range(k, n):
            for j in range(k, n):
                H[i][j] = block[i - k][j - k]
        R = matrix_product(H, R)
        Q = matrix_product(Q, H)
    return Q, R


def column_tail_norm(
        A: list[list[float]],
        j: int
) -> float:
    value = 0.0
    for i in range(j + 1, len(A)):
        value = hypot(value, A[i][j])
    return value


def separation_norm(
        A: list[list[float]],
        end: int
) -> float:
    value = 0.0
    for i in range(end, len(A)):
        for j in range(end):
            value = hypot(value, A[i][j])
    return value


def block_eigenvalues(
        A: list[list[float]],
        i: int
) -> tuple[complex, complex]:
    a = A[i][i]
    b = A[i][i + 1]
    c = A[i + 1][i]
    d = A[i + 1][i + 1]
    root = cmath.sqrt((a - d) ** 2 + 4 * b * c)
    return ((a + d + root) / 2, (a + d - root) / 2)


def pair_difference(
        values: tuple[complex, complex],
        previous: tuple[complex, complex]
) -> float:
    direct = max(abs(values[0] - previous[0]), abs(values[1] - previous[1]))
    reverse = max(abs(values[0] - previous[1]), abs(values[1] - previous[0]))
    return min(direct, reverse)


def isolated_eigenvalues(
        A: list[list[float]],
        A_prev: list[list[float]] | None,
        eps: float
) -> list[float | complex] | None:
    n = len(A)
    lambdas = []
    i = 0

    while i < n:
        tail = column_tail_norm(A, i)
        coupling = separation_norm(A, i + 1)
        real_value = tail <= eps and coupling <= eps
        if real_value and coupling == 0:
            lambdas.append(A[i][i])
            i += 1
            continue

        if i + 1 < n and A_prev is not None:
            if separation_norm(A, i + 2) <= eps:
                values = block_eigenvalues(A, i)
                previous = block_eigenvalues(A_prev, i)
                if pair_difference(values, previous) <= eps:
                    lambdas.extend(values)
                    i += 2
                    continue

        if real_value and A_prev is not None:
            if abs(A[i][i] - A_prev[i][i]) <= eps:
                lambdas.append(A[i][i])
                i += 1
                continue
        return None

    return lambdas


def qr_eigenvalues(
        A0: list[list[float]],
        eps: float,
        max_iter: int = 10000
) -> tuple[list[float | complex], int]:
    A = deepcopy(A0)
    lambdas = isolated_eigenvalues(A, None, eps)
    if lambdas is not None:
        return lambdas, 0

    for k in range(1, max_iter + 1):
        A_prev = A
        Q, R = qr_decomposition(A)
        A = matrix_product(R, Q)
        if not all(isfinite(value) for row in A for value in row):
            raise ValueError("QR-итерации привели к неконечным значениям.")

        lambdas = isolated_eigenvalues(A, A_prev, eps)
        if lambdas is not None:
            return lambdas, k

    raise ValueError(f"QR-алгоритм не сошёлся за {max_iter} итераций.")


def print_problem(
        title: str,
        matrix: list[list[float]],
        eps: float
) -> list[float | complex]:
    print(f"\n{title}")
    Q, R = qr_decomposition(matrix)
    QR = matrix_product(Q, R)
    QQ = matrix_product(transpose(Q), Q)
    print_matrix("A", matrix)
    print_matrix("Q", Q)
    print_matrix("R", R)
    print_matrix("Q · R", QR)
    print(f"Q · R = A: {matrices_equal(QR, matrix)}")
    print(f"Qᵀ · Q = E: {matrices_equal(QQ, identity_matrix(len(matrix)))}")
    print(f"Точность ε = {format_number(eps)}")
    lambdas, k = qr_eigenvalues(matrix, eps)
    print(f"Число QR-итераций: {k}")
    print_vector("Собственные значения", lambdas)
    return lambdas


def main() -> None:
    use_utf8_output()
    A, eps, example = read_problem(input_path("task5.txt"))
    print_problem("1. Основная матрица", A, eps)
    lambdas = print_problem("2. Пример с комплексной парой", example, eps)
    numpy_values = np.linalg.eigvals(np.array(example))
    print_vector("NumPy check", [complex(value) for value in numpy_values])

if __name__ == "__main__":
    main()
