import sys
from math import isfinite
from pathlib import Path


def input_path(
        default_name: str
) -> str:
    if len(sys.argv) == 1:
        return str(Path(__file__).resolve().parent.parent / "data" / default_name)
    if len(sys.argv) == 2:
        return sys.argv[1]
    raise ValueError("Запуск: python <program.py> [input.txt]")


def read_tokens(
        path: str
) -> list[str]:
    with open(path, encoding="utf-8") as source:
        tokens = source.read().split()
    if not tokens:
        raise ValueError("Файл данных пуст.")
    return tokens


def read_size(
        token: str
) -> int:
    n = int(token)
    if n < 1:
        raise ValueError("Размерность должна быть положительной.")
    return n


def read_number(
        token: str
) -> float:
    value = float(token)
    if not isfinite(value):
        raise ValueError("Входные числа должны быть конечными.")
    return value


def read_eps(
        token: str
) -> float:
    eps = read_number(token)
    if eps <= 0:
        raise ValueError("Точность ε должна быть положительной.")
    return eps


def read_vector(
        tokens: list[str],
        n: int,
        start: int
) -> tuple[list[float], int]:
    end = start + n
    if end > len(tokens):
        raise ValueError("Недостаточно элементов в файле данных.")
    values = []
    for token in tokens[start:end]:
        values.append(read_number(token))
    return values, end


def read_matrix(
        tokens: list[str],
        n: int,
        start: int
) -> tuple[list[list[float]], int]:
    A = []
    for i in range(n):
        row, start = read_vector(tokens, n, start)
        A.append(row)
    return A, start


def require_end(
        tokens: list[str],
        end: int
) -> None:
    if end != len(tokens):
        raise ValueError("Лишние данные в конце файла.")


def require_token(
        tokens: list[str],
        start: int
) -> str:
    if start >= len(tokens):
        raise ValueError("Недостаточно данных в файле.")
    return tokens[start]


def format_number(
        value: float | complex
) -> str:
    if isinstance(value, complex):
        if value.imag == 0:
            return f"{value.real:.10g}"
        sign = "+" if value.imag >= 0 else "−"
        return f"{value.real:.10g} {sign} {abs(value.imag):.10g}i"
    return f"{value:.10g}"


def print_rows(
        rows: list[list[str]]
) -> None:
    widths = [0] * len(rows[0])
    for row in rows:
        for j in range(len(row)):
            widths[j] = max(widths[j], len(row[j]))
    for row in rows:
        cells = []
        for j in range(len(row)):
            cells.append(row[j].rjust(widths[j]))
        print("  " + "  ".join(cells))


def print_table(
        headers: tuple[str, ...],
        rows: list[tuple]
) -> None:
    values = [list(headers)]
    for row in rows:
        values.append([str(value) for value in row])
    print_rows(values)


def print_matrix(
        name: str,
        A: list[list[float]]
) -> None:
    print(f"{name}:")
    rows = []
    for row in A:
        rows.append([format_number(value) for value in row])
    print_rows(rows)


def print_vector(
        name: str,
        x: list[float | complex]
) -> None:
    values = [format_number(value) for value in x]
    print(f"{name} = (" + ", ".join(values) + ")")


def save_step(
        trace: list,
        step: tuple
) -> None:
    if len(trace) < 3:
        trace.append(step)
    else:
        trace[-1] = step
