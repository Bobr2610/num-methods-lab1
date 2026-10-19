from math import isfinite
from typing import Callable

from matrix_tools import (
    format_number, input_path, matrix_norm, matrix_product,
    print_vector, read_eps, read_matrix, read_size, read_tokens, read_vector,
    require_end, require_token, use_utf8_output, vector_difference,
)


def read_problem(
        path: str
) -> tuple[list[list[float]], list[float], float, list[float]]:
    tokens = read_tokens(path)
    n = read_size(tokens[0])
    A, start = read_matrix(tokens, n, 1)
    b, start = read_vector(tokens, n, start)
    eps = read_eps(require_token(tokens, start))
    x_ref, end = read_vector(tokens, n, start + 1)
    require_end(tokens, end)
    return A, b, eps, x_ref


def iteration_form(
        A: list[list[float]],
        b: list[float]
) -> tuple[list[list[float]], list[float]]:
    n = len(A)
    alpha = [[0.0] * n for _ in range(n)]
    beta = [0.0] * n
    for i in range(n):
        if A[i][i] == 0:
            raise ValueError("Для приведения системы нужны ненулевые aᵢᵢ.")
        beta[i] = b[i] / A[i][i]
        for j in range(n):
            if i != j:
                alpha[i][j] = -A[i][j] / A[i][i]
    return alpha, beta


def jacobi_iteration(
        alpha: list[list[float]],
        beta: list[float],
        x_prev: list[float]
) -> list[float]:
    n = len(beta)
    x = beta.copy()
    for i in range(n):
        for j in range(n):
            x[i] += alpha[i][j] * x_prev[j]
    return x


def seidel_iteration(
        alpha: list[list[float]],
        beta: list[float],
        x_prev: list[float]
) -> list[float]:
    n = len(beta)
    x = beta.copy()
    for i in range(n):
        for j in range(i):
            x[i] += alpha[i][j] * x[j]
        for j in range(i, n):
            x[i] += alpha[i][j] * x_prev[j]
    return x


def residual_error_factor(
        alpha: list[list[float]],
        max_iter: int
) -> float:
    power = alpha
    norm_sum = 1.0
    for _ in range(max_iter):
        q = matrix_norm(power)
        if not isfinite(q) or not isfinite(norm_sum):
            break
        if q < 1:
            return norm_sum / (1 - q)
        norm_sum += q
        power = matrix_product(power, alpha)
    raise ValueError("Не удалось получить оценку ошибки через степень α.")


def solve_iterations(
        alpha: list[list[float]],
        beta: list[float],
        eps: float,
        method: Callable[[list[list[float]], list[float], list[float]], list[float]],
        max_iter: int = 100000
) -> tuple[list[float], int]:
    q = matrix_norm(alpha)
    if q < 1:
        if method == jacobi_iteration:
            factor = q / (1 - q)
        else:
            C_norm = max(
                sum(abs(alpha[i][j]) for j in range(i, len(alpha)))
                for i in range(len(alpha))
            )
            factor = C_norm / (1 - q)
    else:
        factor = residual_error_factor(alpha, max_iter)

    x = beta.copy()
    for k in range(1, max_iter + 1):
        x_prev = x
        x = method(alpha, beta, x_prev)
        if not all(isfinite(value) for value in x):
            raise ValueError("Итерации расходятся: получены неконечные значения.")

        if q < 1:
            eps_k = factor * vector_difference(x, x_prev)
        else:
            residual = vector_difference(jacobi_iteration(alpha, beta, x), x)
            eps_k = factor * residual
        if eps_k <= eps:
            return x, k

    raise ValueError(f"Метод не сошёлся за {max_iter} итераций.")


def main() -> None:
    use_utf8_output()
    A, b, eps, x_ref = read_problem(input_path("task3.txt"))
    alpha, beta = iteration_form(A, b)

    jacobi = solve_iterations(alpha, beta, eps, jacobi_iteration)
    seidel = solve_iterations(alpha, beta, eps, seidel_iteration)

    print(f"Точность ε = {format_number(eps)}")
    print(f"Простые итерации: {jacobi[1]}")
    print_vector("x простых итераций", jacobi[0])
    print(f"Отклонение от x_ref: {format_number(vector_difference(jacobi[0], x_ref))}")
    print(f"Метод Зейделя: {seidel[1]}")
    print_vector("x Зейделя", seidel[0])
    print(f"Отклонение от x_ref: {format_number(vector_difference(seidel[0], x_ref))}")
    print_vector("Эталон x_ref", x_ref)
    if jacobi[1] > seidel[1]:
        print(f"Зейдель сходится быстрее на {jacobi[1] - seidel[1]} итераций.")
    elif seidel[1] > jacobi[1]:
        print(f"Простые итерации сходятся быстрее на {seidel[1] - jacobi[1]} итераций.")
    else:
        print("Методы сошлись за одинаковое число итераций.")


if __name__ == "__main__":
    main()
