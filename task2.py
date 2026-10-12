from matrix_tools import (
    input_path, print_vector, read_size, read_tokens, read_vector, require_end,
)


def read_system(
        path: str
) -> tuple[list[float], list[float], list[float], list[float]]:
    tokens = read_tokens(path)
    n = read_size(tokens[0])
    a, start = read_vector(tokens, n, 1)
    b, start = read_vector(tokens, n, start)
    c, start = read_vector(tokens, n, start)
    d, end = read_vector(tokens, n, start)
    require_end(tokens, end)
    if a[0] != 0 or c[-1] != 0:
        raise ValueError("Граничные элементы должны быть нулевыми: a₁ = cₙ = 0.")
    return a, b, c, d


def sweep_coefficients(
        a: list[float],
        b: list[float],
        c: list[float],
        d: list[float]
) -> tuple[list[float], list[float]]:
    n = len(b)
    P = [0.0] * n
    Q = [0.0] * n
    if b[0] == 0:
        raise ValueError("Нулевой знаменатель прогонки в строке 1.")
    P[0] = -c[0] / b[0]
    Q[0] = d[0] / b[0]

    for i in range(1, n):
        denom = b[i] + a[i] * P[i - 1]
        if denom == 0:
            raise ValueError(f"Нулевой знаменатель прогонки в строке {i + 1}.")
        P[i] = -c[i] / denom
        Q[i] = (d[i] - a[i] * Q[i - 1]) / denom

    P[-1] = 0.0
    return P, Q


def back_substitution(
        P: list[float],
        Q: list[float]
) -> list[float]:
    n = len(Q)
    x = [0.0] * n
    x[-1] = Q[-1]
    for i in range(n - 2, -1, -1):
        x[i] = P[i] * x[i + 1] + Q[i]
    return x


def residual(
        a: list[float],
        b: list[float],
        c: list[float],
        d: list[float],
        x: list[float]
) -> float:
    error = 0.0
    for i in range(len(x)):
        value = b[i] * x[i] - d[i]
        if i > 0:
            value += a[i] * x[i - 1]
        if i + 1 < len(x):
            value += c[i] * x[i + 1]
        error = max(error, abs(value))
    return error


def main() -> None:
    a, b, c, d = read_system(input_path("task2.txt"))
    P, Q = sweep_coefficients(a, b, c, d)
    x = back_substitution(P, Q)

    print_vector("P", P)
    print_vector("Q", Q)
    print_vector("x", x)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, OverflowError) as error:
        raise SystemExit(str(error))
