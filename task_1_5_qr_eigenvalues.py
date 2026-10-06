from cmath import sqrt

from numerics import input_path, print_matrix, print_values, read_matrix, read_values, require_finite, require_size


def qr_factorization(matrix):
    size = len(matrix)
    upper = [[complex(value) for value in row] for row in matrix]
    orthogonal = [[complex(i == j) for j in range(size)] for i in range(size)]

    for column in range(size):
        vector = [upper[row][column] for row in range(column, size)]
        length = sum(abs(value) ** 2 for value in vector) ** 0.5
        if length == 0:
            continue
        first = vector[0]
        phase = first / abs(first) if first else 1.0 + 0.0j
        vector[0] += phase * length
        denominator = sum(abs(value) ** 2 for value in vector)
        if denominator == 0:
            continue

        for j in range(column, size):
            projection = 2 * sum(vector[k].conjugate() * upper[column + k][j] for k in range(len(vector))) / denominator
            for k, value in enumerate(vector):
                upper[column + k][j] -= value * projection
        for i in range(size):
            projection = 2 * sum(orthogonal[i][column + k] * vector[k] for k in range(len(vector))) / denominator
            for k, value in enumerate(vector):
                orthogonal[i][column + k] -= projection * value.conjugate()

    return orthogonal, upper


def shifted_eigenvalues(matrix, tolerance):
    work = [[complex(value) for value in row] for row in matrix]
    size = len(work)
    active = size
    steps = 0
    limit = max(1000, 5000 * size)

    while active > 1:
        scale = abs(work[active - 1][active - 1]) + abs(work[active - 2][active - 2]) + 1.0
        if abs(work[active - 1][active - 2]) <= tolerance * scale:
            work[active - 1][active - 2] = 0j
            active -= 1
            continue
        if steps >= limit:
            raise ValueError("The shifted QR iteration did not converge within its iteration limit.")

        a = work[active - 2][active - 2]
        b = work[active - 2][active - 1]
        c = work[active - 1][active - 2]
        d = work[active - 1][active - 1]
        root = sqrt((a - d) * (a - d) + 4 * b * c)
        shifts = ((a + d + root) / 2, (a + d - root) / 2)
        shift = min(shifts, key=lambda value: abs(value - d))

        block = [[work[i][j] - (shift if i == j else 0) for j in range(active)] for i in range(active)]
        orthogonal, upper = qr_factorization(block)
        updated = [[sum(upper[i][k] * orthogonal[k][j] for k in range(active)) + (shift if i == j else 0) for j in range(active)] for i in range(active)]
        for i in range(active):
            for j in range(active):
                work[i][j] = updated[i][j]
        steps += 1

    return [work[i][i] for i in range(size)], steps


def main():
    values = read_values(input_path())
    size = int(values[0])
    require_size(size)
    matrix, offset = read_matrix(values, size, 1)
    if len(values) <= offset + 1:
        raise ValueError("Expected a tolerance, demonstration dimension and demonstration matrix.")
    tolerance = float(values[offset])
    example_size = int(values[offset + 1])
    require_size(example_size)
    example, end = read_matrix(values, example_size, offset + 2)
    if end != len(values):
        raise ValueError("The input contains unexpected values after the demonstration matrix.")
    if tolerance <= 0:
        raise ValueError("The tolerance must be positive.")
    require_finite([value for row in matrix for value in row] + [value for row in example for value in row] + [tolerance])

    orthogonal, upper = qr_factorization(matrix)
    eigenvalues, steps = shifted_eigenvalues(matrix, tolerance)
    example_values, example_steps = shifted_eigenvalues(example, tolerance)
    print_matrix("A", matrix)
    print_matrix("Q", orthogonal)
    print_matrix("R", upper)
    print(f"max |Q · R - A|: {max(abs(sum(orthogonal[i][k] * upper[k][j] for k in range(size)) - matrix[i][j]) for i in range(size) for j in range(size)):.3e}")
    print_values("Eigenvalues", eigenvalues)
    print(f"QR iterations: {steps}")
    print_matrix("Complex-pair example", example)
    print_values("Example eigenvalues", example_values)
    print(f"Example QR iterations: {example_steps}")


if __name__ == "__main__":
    main()
