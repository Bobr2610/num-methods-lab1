import math

from input_output import (
    format_number, input_path, print_matrix, print_vector, read_eps,
    read_matrix, read_size, read_tokens, read_vector, require_end,
    require_token, save_step,
)
from matrix_operations import (
    matrix_norm, matrix_vector_product, vector_difference, zero_matrix,
)


def read_problem(path):
    tokens = read_tokens(path)
    n = read_size(tokens[0])
    A, start = read_matrix(tokens, n, 1)
    b, start = read_vector(tokens, n, start)
    eps = read_eps(require_token(tokens, start))
    x_ref, end = read_vector(tokens, n, start + 1)
    require_end(tokens, end)
    return A, b, eps, x_ref


def iteration_form(A, b):
    n = len(A)
    alpha = zero_matrix(n)
    beta = [0.0] * n
    for i in range(n):
        if A[i][i] == 0:
            raise ValueError("Для приведения системы нужны ненулевые aᵢᵢ.")
        beta[i] = b[i] / A[i][i]
        for j in range(n):
            if i != j:
                alpha[i][j] = -A[i][j] / A[i][i]
    return alpha, beta


def split_matrix(alpha):
    n = len(alpha)
    B = zero_matrix(n)
    C = zero_matrix(n)
    for i in range(n):
        for j in range(n):
            if j < i:
                B[i][j] = alpha[i][j]
            else:
                C[i][j] = alpha[i][j]
    return B, C


def jacobi_iteration(alpha, beta, x_prev):
    n = len(beta)
    x = beta.copy()
    for i in range(n):
        for j in range(n):
            x[i] += alpha[i][j] * x_prev[j]
    return x


def seidel_iteration(alpha, beta, x_prev):
    n = len(beta)
    x = beta.copy()
    for i in range(n):
        for j in range(i):
            x[i] += alpha[i][j] * x[j]
        for j in range(i, n):
            x[i] += alpha[i][j] * x_prev[j]
    return x


def solve_iterations(alpha, beta, eps, method, max_iter=100000):
    q = matrix_norm(alpha)
    B, C = split_matrix(alpha)
    factor = 1.0
    if q < 1:
        if method == jacobi_iteration:
            factor = q / (1 - q)
        else:
            factor = matrix_norm(C) / (1 - q)

    x = beta.copy()
    trace = []
    for k in range(1, max_iter + 1):
        x_prev = x
        x = method(alpha, beta, x_prev)
        if not all(math.isfinite(value) for value in x):
            raise ValueError("Итерации расходятся: получены неконечные значения.")

        change = vector_difference(x, x_prev)
        eps_k = factor * change
        save_step(trace, (k, x.copy(), change, eps_k))
        if eps_k <= eps:
            return x, k, trace

    raise ValueError(f"Метод не сошёлся за {max_iter} итераций.")


def print_method(name, result, A, b, x_ref, q, eps):
    x, k, trace = result
    print(f"\n{name}")
    for step, values, change, eps_k in trace:
        print(f"\nИтерация {step}")
        print_vector("x", values)
        print(f"||x − x_prev||c = {format_number(change)}")
        if q < 1:
            print(f"ε⁽ᵏ⁾ = {format_number(eps_k)}; ε⁽ᵏ⁾ ≤ ε: {eps_k <= eps}")
        else:
            print(f"Изменение ≤ ε: {change <= eps}")

    if q >= 1:
        print("\nОстановка по изменению приближений; оценка погрешности не гарантирована.")
    print(f"\nЧисло итераций: {k}")
    print_vector("x", x)
    Ax = matrix_vector_product(A, x)
    print(f"||Ax − b||c = {vector_difference(Ax, b):.3e}")
    print(f"||x − x_ref||c = {vector_difference(x, x_ref):.3e}")


def main():
    A, b, eps, x_ref = read_problem(input_path("task3.txt"))
    alpha, beta = iteration_form(A, b)
    B, C = split_matrix(alpha)
    q = matrix_norm(alpha)

    print_matrix("A", A)
    print_vector("b", b)
    print(f"\nТочность ε = {format_number(eps)}")
    print_matrix("alpha", alpha)
    print_vector("beta", beta)
    print_matrix("B", B)
    print_matrix("C", C)
    print_vector("x⁽⁰⁾ = beta", beta)
    print(f"||alpha||c = {format_number(q)}")
    print(f"||C||c = {format_number(matrix_norm(C))}")

    if q < 1:
        print("Простые итерации: ε⁽ᵏ⁾ = ||alpha||c / (1 − ||alpha||c) · ||x − x_prev||c")
        print("Зейдель: ε⁽ᵏ⁾ = ||C||c / (1 − ||alpha||c) · ||x − x_prev||c")
    else:
        print("Критерий остановки: ||x − x_prev||c ≤ ε")

    print_vector("Эталон x_ref", x_ref)
    jacobi = solve_iterations(alpha, beta, eps, jacobi_iteration)
    seidel = solve_iterations(alpha, beta, eps, seidel_iteration)
    print_method("1. Простые итерации", jacobi, A, b, x_ref, q, eps)
    print_method("2. Метод Зейделя", seidel, A, b, x_ref, q, eps)

    print("\n3. Сравнение по числу итераций")
    print(f"Простые итерации: {jacobi[1]}; Зейдель: {seidel[1]}")
    if jacobi[1] > seidel[1]:
        print(f"Зейдель потребовал на {jacobi[1] - seidel[1]} итераций меньше.")
    elif seidel[1] > jacobi[1]:
        print(f"Простые итерации потребовали на {seidel[1] - jacobi[1]} итераций меньше.")
    else:
        print("Число итераций одинаково.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, OverflowError) as error:
        raise SystemExit(str(error))
