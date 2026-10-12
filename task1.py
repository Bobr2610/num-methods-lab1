from input_output import (
    format_number, input_path, print_matrix, print_vector, read_matrix,
    read_size, read_tokens, read_vector, require_end, save_step,
)
from matrix_operations import (
    copy_matrix, identity_matrix, matrix_difference, matrix_product,
    matrix_vector_product, vector_difference, zero_matrix,
)


def read_system(path):
    tokens = read_tokens(path)
    n = read_size(tokens[0])
    A, start = read_matrix(tokens, n, 1)
    b, end = read_vector(tokens, n, start)
    require_end(tokens, end)
    return A, b


def lu_decomposition(A):
    n = len(A)
    L = identity_matrix(n)
    U = copy_matrix(A)
    P = identity_matrix(n)
    p = 0
    trace = []

    for k in range(n):
        m = k
        for i in range(k + 1, n):
            if abs(U[i][k]) > abs(U[m][k]):
                m = i
        if U[m][k] == 0:
            raise ValueError("Матрица вырождена: нулевой главный элемент.")

        if m != k:
            U[k], U[m] = U[m], U[k]
            P[k], P[m] = P[m], P[k]
            for j in range(k):
                L[k][j], L[m][j] = L[m][j], L[k][j]
            p += 1

        values = []
        for i in range(k + 1, n):
            mu = U[i][k] / U[k][k]
            L[i][k] = mu
            values.append((i + 1, mu))
            for j in range(k + 1, n):
                U[i][j] -= mu * U[k][j]
            U[i][k] = 0.0

        save_step(trace, (k + 1, m + 1, values, copy_matrix(U)))

    return L, U, P, p, trace


def forward_substitution(L, b):
    n = len(L)
    z = [0.0] * n
    for i in range(n):
        z[i] = b[i]
        for j in range(i):
            z[i] -= L[i][j] * z[j]
    return z


def back_substitution(U, z):
    n = len(U)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = z[i]
        for j in range(i + 1, n):
            x[i] -= U[i][j] * x[j]
        x[i] /= U[i][i]
    return x


def solve_lu(L, U, P, b):
    Pb = matrix_vector_product(P, b)
    z = forward_substitution(L, Pb)
    x = back_substitution(U, z)
    return Pb, z, x


def inverse_from_lu(L, U, P):
    n = len(L)
    E = identity_matrix(n)
    X = zero_matrix(n)
    for j in range(n):
        b = [E[i][j] for i in range(n)]
        Pb, z, x = solve_lu(L, U, P, b)
        for i in range(n):
            X[i][j] = x[i]
    return X


def determinant_from_lu(U, p):
    det = (-1.0) ** p
    for i in range(len(U)):
        det *= U[i][i]
    return det


def print_elimination(trace):
    print("\n1. Исключение с выбором главного элемента")
    for k, m, values, U in trace:
        print(f"\nШаг {k}: выбрана строка {m}")
        for i, mu in values:
            print(f"μ[{i}, {k}] = {format_number(mu)}")
        print_matrix("U", U)


def main():
    A, b = read_system(input_path("task1.txt"))
    L, U, P, p, trace = lu_decomposition(A)
    print_matrix("A", A)
    print_vector("b", b)
    print_elimination(trace)

    print("\n2. Разложение PA = LU")
    LU = matrix_product(L, U)
    PA = matrix_product(P, A)
    print_matrix("P", P)
    print_matrix("L", L)
    print_matrix("U", U)
    print_matrix("L · U", LU)
    print_matrix("P · A", PA)
    print(f"max |LU − PA| = {matrix_difference(LU, PA):.3e}")

    print("\n3. Решение Lz = Pb, Ux = z")
    Pb, z, x = solve_lu(L, U, P, b)
    print_vector("Pb", Pb)
    print_vector("z", z)
    print_vector("x", x)
    Ax = matrix_vector_product(A, x)
    print(f"max |Ax − b| = {vector_difference(Ax, b):.3e}")

    print("\n4. Обратная матрица и определитель")
    X = inverse_from_lu(L, U, P)
    det = determinant_from_lu(U, p)
    print_matrix("X = A⁻¹", X)
    print(f"Перестановок p = {p}")
    print(f"det A = (−1)^p · ∏ uᵢᵢ = {format_number(det)}")

    print("\n5. Проверка AX = E")
    AX = matrix_product(A, X)
    E = identity_matrix(len(A))
    print_matrix("A · X", AX)
    print_matrix("E", E)
    print(f"max |AX − E| = {matrix_difference(AX, E):.3e}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, OverflowError) as error:
        raise SystemExit(str(error))
