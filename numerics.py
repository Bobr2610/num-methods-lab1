import math


def read_values(path):
    with open(path, encoding="utf-8") as source:
        return source.read().split()


def read_matrix(values, size, offset):
    end = offset + size * size
    matrix = [list(map(float, values[i:i + size])) for i in range(offset, end, size)]
    if len(matrix) != size or any(len(row) != size for row in matrix):
        raise ValueError("The input does not contain a complete square matrix.")
    return matrix, end


def identity(size):
    return [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]


def transpose(matrix):
    return [list(column) for column in zip(*matrix)]


def multiply(left, right):
    columns = transpose(right)
    return [[sum(a * b for a, b in zip(row, column)) for column in columns] for row in left]


def matrix_vector(matrix, vector):
    return [sum(value * item for value, item in zip(row, vector)) for row in matrix]


def infinity_norm(values):
    return max((abs(value) for value in values), default=0.0)


def print_matrix(name, matrix):
    print(f"{name}:")
    for row in matrix:
        print("  " + "  ".join(format_number(value) for value in row))


def print_vector(name, vector):
    print(f"{name}: [" + ", ".join(f"{value: .10g}" for value in vector) + "]")


def format_number(value):
    if isinstance(value, complex):
        if abs(value.imag) < 1e-12:
            return f"{value.real:.10g}"
        sign = "+" if value.imag >= 0 else "-"
        return f"{value.real:.10g} {sign} {abs(value.imag):.10g}i"
    return f"{value:.10g}"


def print_values(name, values):
    print(f"{name}: [" + ", ".join(format_number(value) for value in values) + "]")


def require_size(size):
    if size < 1:
        raise ValueError("The matrix dimension must be positive.")


def input_path():
    import sys

    if len(sys.argv) != 2:
        raise SystemExit("Usage: python <program.py> <input-file>")
    return sys.argv[1]


def require_finite(values):
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("All input values must be finite numbers.")
