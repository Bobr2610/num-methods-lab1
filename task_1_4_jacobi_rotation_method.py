import math

from input_output import (
    format_number, input_path, print_matrix, print_vector, read_eps,
    read_matrix, read_size, read_tokens, require_end, require_token, save_step,
)
from matrix_operations import (
    copy_matrix, identity_matrix, matrix_difference, matrix_product,
    off_diagonal_norm, transpose, zero_matrix,
)


def read_problem(path):
    tokens = read_tokens(path)
    n = read_size(tokens[0])
    A, start = read_matrix(tokens, n, 1)
    eps = read_eps(require_token(tokens, start))
    require_end(tokens, start + 1)

    for i in range(n):
        for j in range(i + 1, n):
            if not math.isclose(A[i][j], A[j][i], rel_tol=1e-12, abs_tol=1e-14):
                raise ValueError("Метод вращений требует симметричную матрицу.")
    return A, eps


def largest_off_diagonal(A):
    i, j = 0, 0
    value = 0.0
    for row in range(len(A)):
        for column in range(row + 1, len(A)):
            if abs(A[row][column]) > value:
                i, j = row, column
                value = abs(A[row][column])
    return i, j


def jacobi_rotation(A, i, j):
    if A[i][i] == A[j][j]:
        phi = math.pi / 4
    else:
        phi = 0.5 * math.atan(2 * A[i][j] / (A[i][i] - A[j][j]))

    U = identity_matrix(len(A))
    U[i][i] = math.cos(phi)
    U[j][j] = math.cos(phi)
    U[i][j] = -math.sin(phi)
    U[j][i] = math.sin(phi)
    return U, phi


def rotation_method(A0, eps, max_iter=100000):
    A = copy_matrix(A0)
    V = identity_matrix(len(A))
    trace = []
    t = off_diagonal_norm(A)
    k = 0

    while t >= eps:
        if k == max_iter:
            raise ValueError(f"Метод вращений не сошёлся за {max_iter} шагов.")
        i, j = largest_off_diagonal(A)
        value = A[i][j]
        U, phi = jacobi_rotation(A, i, j)
        A = matrix_product(matrix_product(transpose(U), A), U)
        V = matrix_product(V, U)
        t = off_diagonal_norm(A)
        if not math.isfinite(t):
            raise ValueError("Получены неконечные значения при вращении.")
        k += 1
        save_step(trace, (k, i, j, value, phi, U, copy_matrix(A), t))

    return A, V, k, trace


def print_rotations(trace, eps):
    for k, i, j, value, phi, U, A, t in trace:
        print(f"\nВращение {k}: i = {i + 1}, j = {j + 1}")
        print(f"aᵢⱼ = {format_number(value)}; φ = {format_number(phi)} рад")
        print_matrix("U", U)
        print_matrix("A после вращения", A)
        print(f"t(A) = {format_number(t)}; t(A) < ε: {t < eps}")


def main():
    A0, eps = read_problem(input_path())
    print_matrix("A⁽⁰⁾", A0)
    print(f"\nТочность ε = {format_number(eps)}")
    print("Критерий: t(A) = sqrt(Σᵢ<ⱼ aᵢⱼ²) < ε")
    print(f"Начальное t(A) = {format_number(off_diagonal_norm(A0))}")

    A, V, k, trace = rotation_method(A0, eps)
    print_rotations(trace, eps)

    print("\nРезультат")
    n = len(A)
    lambdas = [A[i][i] for i in range(n)]
    Lambda = zero_matrix(n)
    for i in range(n):
        Lambda[i][i] = lambdas[i]
    print(f"Число вращений: {k}")
    print(f"Конечное t(A) = {format_number(off_diagonal_norm(A))}")
    print_vector("Собственные значения", lambdas)
    print_matrix("V, собственные векторы по столбцам", V)
    print_matrix("Λ", Lambda)

    print("\nПроверки")
    AV = matrix_product(A0, V)
    VL = matrix_product(V, Lambda)
    VV = matrix_product(transpose(V), V)
    print_matrix("A⁽⁰⁾ · V", AV)
    print_matrix("V · Λ", VL)
    print(f"max |AV − VΛ| = {matrix_difference(AV, VL):.3e}")
    print(f"max |VᵀV − E| = {matrix_difference(VV, identity_matrix(n)):.3e}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, OverflowError) as error:
        raise SystemExit(str(error))
