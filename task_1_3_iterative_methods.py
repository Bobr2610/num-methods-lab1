from numerics import infinity_norm, input_path, matrix_vector, print_vector, read_matrix, read_values, require_finite, require_size


def iterate(matrix, right, tolerance, reference, seidel):
    size = len(matrix)
    for i, row in enumerate(matrix):
        if abs(row[i]) <= 1e-14:
            raise ValueError(f"The diagonal entry in row {i + 1} is zero.")
        if sum(abs(row[j]) for j in range(size) if j != i) >= abs(row[i]):
            raise ValueError("The supplied matrix is not strictly diagonally dominant; convergence is not guaranteed.")

    solution = [0.0] * size
    for iteration in range(1, 1_000_001):
        previous = solution[:]
        for i, row in enumerate(matrix):
            current = solution if seidel else previous
            left = sum(row[j] * current[j] for j in range(i))
            right_sum = sum(row[j] * previous[j] for j in range(i + 1, size))
            solution[i] = (right[i] - left - right_sum) / row[i]
        if infinity_norm([solution[i] - previous[i] for i in range(size)]) <= tolerance:
            residual = infinity_norm([a - b for a, b in zip(matrix_vector(matrix, solution), right)])
            error = infinity_norm([a - b for a, b in zip(solution, reference)])
            return solution, iteration, residual, error
    raise ValueError("The iteration limit was reached before meeting the requested tolerance.")


def main():
    values = read_values(input_path())
    size = int(values[0])
    require_size(size)
    matrix, offset = read_matrix(values, size, 1)
    right = list(map(float, values[offset:offset + size]))
    tolerance_index = offset + size
    if len(values) != tolerance_index + size + 1:
        raise ValueError("Expected n matrix rows, the right-hand side, a tolerance and a reference solution.")
    tolerance = float(values[tolerance_index])
    reference = list(map(float, values[tolerance_index + 1:]))
    if tolerance <= 0:
        raise ValueError("The tolerance must be positive.")
    require_finite([value for row in matrix for value in row] + right + reference + [tolerance])

    jacobi, jacobi_count, jacobi_residual, jacobi_error = iterate(matrix, right, tolerance, reference, False)
    seidel, seidel_count, seidel_residual, seidel_error = iterate(matrix, right, tolerance, reference, True)
    print(f"Tolerance ε: {tolerance:.10g}")
    print_vector("Reference solution", reference)
    print_vector("Jacobi solution", jacobi)
    print(f"Jacobi iterations: {jacobi_count}")
    print(f"Jacobi residual ||Ax - b||∞: {jacobi_residual:.3e}")
    print(f"Jacobi reference error ||x - x_ref||∞: {jacobi_error:.3e}")
    print_vector("Gauss–Seidel solution", seidel)
    print(f"Gauss–Seidel iterations: {seidel_count}")
    print(f"Gauss–Seidel residual ||Ax - b||∞: {seidel_residual:.3e}")
    print(f"Gauss–Seidel reference error ||x - x_ref||∞: {seidel_error:.3e}")
    if jacobi_count > seidel_count:
        print(f"Gauss–Seidel used {jacobi_count - seidel_count} fewer iterations.")
    elif seidel_count > jacobi_count:
        print(f"Jacobi used {seidel_count - jacobi_count} fewer iterations.")
    else:
        print("Both methods used the same number of iterations.")


if __name__ == "__main__":
    main()
