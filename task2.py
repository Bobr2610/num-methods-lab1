from input_output import (
    format_number, input_path, print_table, print_vector, read_size,
    read_tokens, read_vector, require_end,
)


def read_system(path):
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


def sweep_coefficients(a, b, c, d):
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


def back_substitution(P, Q):
    n = len(Q)
    x = [0.0] * n
    x[-1] = Q[-1]
    for i in range(n - 2, -1, -1):
        x[i] = P[i] * x[i + 1] + Q[i]
    return x


def residual(a, b, c, d, x):
    error = 0.0
    for i in range(len(x)):
        value = b[i] * x[i] - d[i]
        if i > 0:
            value += a[i] * x[i - 1]
        if i + 1 < len(x):
            value += c[i] * x[i + 1]
        error = max(error, abs(value))
    return error


def print_solution(P, Q, x):
    print("\n1. Прямой ход")
    print("Pᵢ = −cᵢ / (bᵢ + aᵢPᵢ₋₁)")
    print("Qᵢ = (dᵢ − aᵢQᵢ₋₁) / (bᵢ + aᵢPᵢ₋₁)")
    rows = []
    for i in range(len(P)):
        rows.append((i + 1, format_number(P[i]), format_number(Q[i])))
    print_table(("i", "Pᵢ", "Qᵢ"), rows)

    print("\n2. Обратный ход")
    n = len(x)
    print(f"x[{n}] = Q[{n}] = {format_number(x[-1])}")
    for i in range(n - 2, -1, -1):
        terms = f"{format_number(P[i])} · {format_number(x[i + 1])}"
        value = format_number(x[i])
        print(f"x[{i + 1}] = {terms} + ({format_number(Q[i])}) = {value}")
    print_vector("x", x)


def main():
    a, b, c, d = read_system(input_path("task2.txt"))
    print_vector("a", a)
    print_vector("b", b)
    print_vector("c", c)
    print_vector("d", d)
    P, Q = sweep_coefficients(a, b, c, d)
    x = back_substitution(P, Q)
    print_solution(P, Q, x)
    print(f"\nmax невязка = {residual(a, b, c, d, x):.3e}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, OverflowError) as error:
        raise SystemExit(str(error))
