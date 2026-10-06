from numerics import identity, input_path, multiply, print_matrix, print_values, read_matrix, read_values, require_finite, require_size


def diagonalize(matrix, tolerance):
    size = len(matrix)
    diagonal = [row[:] for row in matrix]
    vectors = identity(size)
    limit = max(1, 100 * size * size)

    for _ in range(limit):
        p, q = max(((i, j) for i in range(size) for j in range(i + 1, size)), key=lambda pair: abs(diagonal[pair[0]][pair[1]]), default=(0, 0))
        largest = abs(diagonal[p][q]) if size > 1 else 0.0
        if largest <= tolerance:
            return diagonal, vectors, _

        app = diagonal[p][p]
        aqq = diagonal[q][q]
        apq = diagonal[p][q]
        tau = (aqq - app) / (2.0 * apq)
        sign = 1.0 if tau >= 0 else -1.0
        tangent = sign / (abs(tau) + (1.0 + tau * tau) ** 0.5)
        cosine = 1.0 / (1.0 + tangent * tangent) ** 0.5
        sine = tangent * cosine

        diagonal[p][p] = app - tangent * apq
        diagonal[q][q] = aqq + tangent * apq
        diagonal[p][q] = diagonal[q][p] = 0.0
        for k in range(size):
            if k not in (p, q):
                akp = diagonal[k][p]
                akq = diagonal[k][q]
                diagonal[k][p] = diagonal[p][k] = cosine * akp - sine * akq
                diagonal[k][q] = diagonal[q][k] = sine * akp + cosine * akq
            vkp = vectors[k][p]
            vkq = vectors[k][q]
            vectors[k][p] = cosine * vkp - sine * vkq
            vectors[k][q] = sine * vkp + cosine * vkq
    raise ValueError("The rotation limit was reached before meeting the requested tolerance.")


def main():
    values = read_values(input_path())
    size = int(values[0])
    require_size(size)
    matrix, offset = read_matrix(values, size, 1)
    if len(values) != offset + 1:
        raise ValueError("Expected n matrix rows followed by the tolerance.")
    tolerance = float(values[offset])
    if tolerance <= 0:
        raise ValueError("The tolerance must be positive.")
    require_finite([value for row in matrix for value in row] + [tolerance])
    if any(abs(matrix[i][j] - matrix[j][i]) > tolerance for i in range(size) for j in range(size)):
        raise ValueError("The Jacobi rotation method requires a symmetric matrix.")

    diagonal, vectors, rotations = diagonalize(matrix, tolerance)
    eigenvalues = [diagonal[i][i] for i in range(size)]
    order = sorted(range(size), key=lambda i: eigenvalues[i])
    eigenvalues = [eigenvalues[i] for i in order]
    vectors = [[row[i] for i in order] for row in vectors]
    eigenvalue_matrix = [[eigenvalues[i] if i == j else 0.0 for j in range(size)] for i in range(size)]
    left = multiply(matrix, vectors)
    right = multiply(vectors, eigenvalue_matrix)
    error = max(abs(left[i][j] - right[i][j]) for i in range(size) for j in range(size))

    print(f"Tolerance ε: {tolerance:.10g}")
    print(f"Rotations: {rotations}")
    print_values("Eigenvalues", eigenvalues)
    print_matrix("Eigenvectors V (columns)", vectors)
    print_matrix("Λ", eigenvalue_matrix)
    print(f"max |A · V - V · Λ|: {error:.3e}")


if __name__ == "__main__":
    main()
