from copy import deepcopy

from matrix_tools import (
    format_number, identity_matrix, input_path, matrix_product,
    matrix_vector_product, matrices_equal, print_matrix,
    print_vector, read_matrix, read_size, read_tokens, read_vector,
    require_end, use_utf8_output, zero_matrix,
)


def read_system(
        path: str
) -> tuple[list[list[float]], list[float]]:
    tokens = read_tokens(path)
    n = read_size(tokens[0])
    A, start = read_matrix(tokens, n, 1)
    b, end = read_vector(tokens, n, start)
    require_end(tokens, end)
    return A, b


def pivot_row(
        U: list[list[float]],
        k: int
) -> int:
    m = k
    for i in range(k + 1, len(U)):
        if abs(U[i][k]) > abs(U[m][k]):
            m = i
    return m


def lu_decomposition(
        A: list[list[float]]
) -> tuple[list[list[float]], list[list[float]], list[list[float]], int]:
    n = len(A)
    L = identity_matrix(n)
    U = deepcopy(A)
    P = identity_matrix(n)
    p = 0
    for k in range(n):
        m = pivot_row(U, k)
        if U[m][k] == 0:
            raise ValueError("Матрица вырождена: нулевой главный элемент.")

        if m != k:
            U[k], U[m] = U[m], U[k]
            P[k], P[m] = P[m], P[k]
            for j in range(k):
                L[k][j], L[m][j] = L[m][j], L[k][j]
            p += 1

        for i in range(k + 1, n):
            mu = U[i][k] / U[k][k]
            L[i][k] = mu
            for j in range(k + 1, n):
                U[i][j] -= mu * U[k][j]
            U[i][k] = 0.0

    return L, U, P, p


def forward_substitution(
        L: list[list[float]],
        b: list[float]
) -> list[float]:
    n = len(L)
    z = [0.0] * n
    for i in range(n):
        z[i] = b[i]
        for j in range(i):
            z[i] -= L[i][j] * z[j]
    return z


def back_substitution(
        U: list[list[float]],
        z: list[float]
) -> list[float]:
    n = len(U)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = z[i]
        for j in range(i + 1, n):
            x[i] -= U[i][j] * x[j]
        x[i] /= U[i][i]
    return x


def solve_lu(
        L: list[list[float]],
        U: list[list[float]],
        P: list[list[float]],
        b: list[float]
) -> list[float]:
    Pb = matrix_vector_product(P, b)
    z = forward_substitution(L, Pb)
    x = back_substitution(U, z)
    return x


def inverse_from_lu(
        L: list[list[float]],
        U: list[list[float]],
        P: list[list[float]]
) -> list[list[float]]:
    n = len(P)
    E = identity_matrix(n)
    X = zero_matrix(n)
    for j in range(n):
        b = [E[i][j] for i in range(n)]
        x = solve_lu(L, U, P, b)
        for i in range(n):
            X[i][j] = x[i]
    return X


def determinant_from_lu(
        U: list[list[float]],
        p: int
) -> float:
    det = (-1.0) ** p
    for i in range(len(U)):
        det *= U[i][i]
    return det


def main() -> None:
    use_utf8_output()
    A, b = read_system(input_path("task1.txt"))
    L, U, P, p = lu_decomposition(A)
    LU = matrix_product(L, U)
    PA = matrix_product(P, A)
    x = solve_lu(L, U, P, b)
    X = inverse_from_lu(L, U, P)
    det = determinant_from_lu(U, p)
    AX = matrix_product(A, X)
    E = identity_matrix(len(A))

    print_matrix("L", L)
    print_matrix("U", U)
    print_matrix("L · U", LU)
    print_vector("x", x)
    print_matrix("A^-1", X)
    print(f"det A = {format_number(det)}")
    print(f"A · A^-1 = E: {matrices_equal(AX, E)}")
    print(f"L · U = P · A: {matrices_equal(LU, PA)}")


if __name__ == "__main__":
    main()
