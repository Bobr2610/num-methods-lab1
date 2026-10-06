from numerics import identity, input_path, matrix_vector, multiply, print_matrix, print_vector, read_matrix, read_values, require_finite, require_size, transpose


def factorize(matrix):
    size = len(matrix)
    upper = [row[:] for row in matrix]
    lower = identity(size)
    permutation = identity(size)
    swaps = 0
    scale = max((abs(value) for row in matrix for value in row), default=0.0)
    threshold = 1e-14 * max(1.0, scale)

    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(upper[row][column]))
        if abs(upper[pivot][column]) <= threshold:
            raise ValueError("The matrix is singular to working precision.")
        if pivot != column:
            upper[column], upper[pivot] = upper[pivot], upper[column]
            permutation[column], permutation[pivot] = permutation[pivot], permutation[column]
            lower[column][:column], lower[pivot][:column] = lower[pivot][:column], lower[column][:column]
            swaps += 1

        for row in range(column + 1, size):
            multiplier = upper[row][column] / upper[column][column]
            lower[row][column] = multiplier
            for j in range(column, size):
                upper[row][j] -= multiplier * upper[column][j]

    return lower, upper, permutation, swaps


def solve(lower, upper, permutation, right):
    permuted = matrix_vector(permutation, right)
    forward = []
    for i, row in enumerate(lower):
        forward.append(permuted[i] - sum(row[j] * forward[j] for j in range(i)))

    solution = [0.0] * len(upper)
    for i in range(len(upper) - 1, -1, -1):
        remainder = forward[i] - sum(upper[i][j] * solution[j] for j in range(i + 1, len(upper)))
        solution[i] = remainder / upper[i][i]
    return solution


def main():
    values = read_values(input_path())
    size = int(values[0])
    require_size(size)
    matrix, offset = read_matrix(values, size, 1)
    right = list(map(float, values[offset:offset + size]))
    if len(right) != size or offset + size != len(values):
        raise ValueError("Expected n matrix rows followed by one right-hand-side vector.")
    require_finite([value for row in matrix for value in row] + right)
    lower, upper, permutation, swaps = factorize(matrix)
    product = multiply(lower, upper)
    inverse = transpose([solve(lower, upper, permutation, [1.0 if i == j else 0.0 for i in range(size)]) for j in range(size)])
    determinant = (-1.0 if swaps % 2 else 1.0)
    for i in range(size):
        determinant *= upper[i][i]

    print_matrix("A", matrix)
    print_matrix("P", permutation)
    print_matrix("L", lower)
    print_matrix("U", upper)
    print_matrix("L · U", product)
    print_vector("Solution x", solve(lower, upper, permutation, right))
    print_matrix("A⁻¹", inverse)
    print(f"det(A): {determinant:.10g}")
    print_matrix("A · A⁻¹", multiply(matrix, inverse))
    permuted_matrix = multiply(permutation, matrix)
    print(f"max |L · U - P · A|: {max(abs(product[i][j] - permuted_matrix[i][j]) for i in range(size) for j in range(size)):.3e}")


if __name__ == "__main__":
    main()
