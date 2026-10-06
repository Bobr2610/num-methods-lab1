from numerics import input_path, print_vector, read_values, require_finite, require_size


def solve_tridiagonal(lower, diagonal, upper, right):
    size = len(diagonal)
    if any(len(vector) != size for vector in (lower, upper, right)):
        raise ValueError("All four input vectors must have length n.")
    if abs(lower[0]) > 1e-14 or abs(upper[-1]) > 1e-14:
        raise ValueError("The first lower and last upper diagonal entries must be zero.")

    alpha = [0.0] * size
    beta = [0.0] * size
    scale = max((abs(value) for value in diagonal), default=0.0)
    threshold = 1e-14 * max(1.0, scale)
    denominator = diagonal[0]
    if abs(denominator) <= threshold:
        raise ValueError("The first Thomas denominator is zero to working precision.")
    alpha[0] = -upper[0] / denominator
    beta[0] = right[0] / denominator

    for i in range(1, size):
        denominator = diagonal[i] + lower[i] * alpha[i - 1]
        if abs(denominator) <= threshold:
            raise ValueError(f"Thomas denominator at row {i + 1} is zero to working precision.")
        alpha[i] = -upper[i] / denominator if i < size - 1 else 0.0
        beta[i] = (right[i] - lower[i] * beta[i - 1]) / denominator

    solution = [0.0] * size
    solution[-1] = beta[-1]
    for i in range(size - 2, -1, -1):
        solution[i] = alpha[i] * solution[i + 1] + beta[i]
    return alpha, beta, solution


def main():
    values = read_values(input_path())
    size = int(values[0])
    require_size(size)
    numbers = list(map(float, values[1:]))
    if len(numbers) != 4 * size:
        raise ValueError("Expected lower, main and upper diagonals, followed by the right-hand side.")
    require_finite(numbers)
    lower, diagonal, upper, right = [numbers[i * size:(i + 1) * size] for i in range(4)]
    alpha, beta, solution = solve_tridiagonal(lower, diagonal, upper, right)
    print_vector("alpha", alpha)
    print_vector("beta", beta)
    print_vector("Solution x", solution)


if __name__ == "__main__":
    main()
