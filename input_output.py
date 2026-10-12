import math
import sys
from pathlib import Path


def input_path(default_name):
    if len(sys.argv) == 1:
        return str(Path(__file__).resolve().parent / "data" / default_name)
    if len(sys.argv) == 2:
        return sys.argv[1]
    raise ValueError("Запуск: python <program.py> [input.txt]")


def read_tokens(path):
    with open(path, encoding="utf-8") as source:
        tokens = source.read().split()
    if not tokens:
        raise ValueError("Файл данных пуст.")
    return tokens


def read_size(token):
    n = int(token)
    if n < 1:
        raise ValueError("Размерность должна быть положительной.")
    return n


def read_number(token):
    value = float(token)
    if not math.isfinite(value):
        raise ValueError("Входные числа должны быть конечными.")
    return value


def read_eps(token):
    eps = read_number(token)
    if eps <= 0:
        raise ValueError("Точность ε должна быть положительной.")
    return eps


def read_vector(tokens, n, start):
    end = start + n
    if end > len(tokens):
        raise ValueError("Недостаточно элементов в файле данных.")
    values = []
    for token in tokens[start:end]:
        values.append(read_number(token))
    return values, end


def read_matrix(tokens, n, start):
    A = []
    for i in range(n):
        row, start = read_vector(tokens, n, start)
        A.append(row)
    return A, start


def require_end(tokens, end):
    if end != len(tokens):
        raise ValueError("Лишние данные в конце файла.")


def require_token(tokens, start):
    if start >= len(tokens):
        raise ValueError("Недостаточно данных в файле.")
    return tokens[start]


def format_number(value):
    if isinstance(value, complex):
        if value.imag == 0:
            return f"{value.real:.10g}"
        sign = "+" if value.imag >= 0 else "−"
        return f"{value.real:.10g} {sign} {abs(value.imag):.10g}i"
    return f"{value:.10g}"


def print_rows(rows):
    widths = [0] * len(rows[0])
    for row in rows:
        for j in range(len(row)):
            widths[j] = max(widths[j], len(row[j]))
    for row in rows:
        cells = []
        for j in range(len(row)):
            cells.append(row[j].rjust(widths[j]))
        print("  " + "  ".join(cells))


def print_table(headers, rows):
    values = [list(headers)]
    for row in rows:
        values.append([str(value) for value in row])
    print_rows(values)


def print_matrix(name, A):
    print(f"{name}:")
    rows = []
    for row in A:
        rows.append([format_number(value) for value in row])
    print_rows(rows)


def print_vector(name, x):
    values = [format_number(value) for value in x]
    print(f"{name} = (" + ", ".join(values) + ")")


def save_step(trace, step):
    if len(trace) < 3:
        trace.append(step)
    else:
        trace[-1] = step
