from copy import deepcopy
from math import atan, cos, isclose, isfinite, pi, sin

from matrix_tools import (
    format_number, identity_matrix, input_path, matrix_product,
    matrices_equal, off_diagonal_norm, print_matrix, print_vector, read_eps,
    read_matrix, read_size, read_tokens, require_end, require_token,
    print_table, transpose, use_utf8_output, zero_matrix,
)


def read_problem(
        path: str
) -> tuple[list[list[float]], float]:
    tokens = read_tokens(path)
    n = read_size(tokens[0])
    A, start = read_matrix(tokens, n, 1)
    eps = read_eps(require_token(tokens, start))
    require_end(tokens, start + 1)

    for i in range(n):
        for j in range(i + 1, n):
            if not isclose(A[i][j], A[j][i], rel_tol=1e-12, abs_tol=1e-14):
                raise ValueError("Метод вращений требует симметричную матрицу.")
    return A, eps


def largest_off_diagonal(
        A: list[list[float]]
) -> tuple[int, int]:
    i, j = 0, 0
    value = 0.0
    for row in range(len(A)):
        for column in range(row + 1, len(A)):
            if abs(A[row][column]) > value:
                i, j = row, column
                value = abs(A[row][column])
    return i, j


def jacobi_rotation(
        A: list[list[float]],
        i: int,
        j: int
) -> tuple[list[list[float]], float]:
    if A[i][i] == A[j][j]:
        phi = pi / 4
    else:
        phi = 0.5 * atan(2 * A[i][j] / (A[i][i] - A[j][j]))

    U = identity_matrix(len(A))
    U[i][i] = cos(phi)
    U[j][j] = cos(phi)
    U[i][j] = -sin(phi)
    U[j][i] = sin(phi)
    return U, phi


def rotation_method(
        A0: list[list[float]],
        eps: float,
        max_iter: int = 100000
) -> tuple[list[list[float]], list[list[float]], int, list[tuple[int, float]]]:
    A = deepcopy(A0)
    V = identity_matrix(len(A))
    t = off_diagonal_norm(A)
    k = 0
    history = [(k, t)]

    while t >= eps:
        if k == max_iter:
            raise ValueError(f"Метод вращений не сошёлся за {max_iter} шагов.")
        i, j = largest_off_diagonal(A)
        U, _ = jacobi_rotation(A, i, j)
        A = matrix_product(matrix_product(transpose(U), A), U)
        V = matrix_product(V, U)
        t = off_diagonal_norm(A)
        if not isfinite(t):
            raise ValueError("Получены неконечные значения при вращении.")
        k += 1
        history.append((k, t))

    return A, V, k, history


def main() -> None:
    use_utf8_output()
    A0, eps = read_problem(input_path("task4.txt"))
    A, V, k, history = rotation_method(A0, eps)
    n = len(A)
    lambdas = [A[i][i] for i in range(n)]
    Lambda = zero_matrix(n)
    for i in range(n):
        Lambda[i][i] = lambdas[i]

    print_matrix("A", A0)
    print(f"Точность ε = {format_number(eps)}")
    print(f"Число вращений: {k}")
    print_table(("Вращение", "Норма внедиагональной части"), history)
    print_vector("Собственные значения", lambdas)
    print_matrix("V, собственные векторы по столбцам", V)
    print(f"A · V = V · Λ: {matrices_equal(matrix_product(A0, V), matrix_product(V, Lambda), eps)}")


if __name__ == "__main__":
    main()
