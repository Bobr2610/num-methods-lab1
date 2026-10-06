import ast
import io
import itertools
import os
from pathlib import Path
import subprocess
import sys
import tokenize
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import input_output
import matrix_operations as matrices
import task_1_1_lu_decomposition as lu
import task_1_2_sweep_method as sweep
import task_1_3_jacobi_and_seidel_methods as iterations
import task_1_4_jacobi_rotation_method as rotations
import task_1_5_qr_eigenvalues as qr


def fixture(name):
    return ROOT / "tests" / "data" / name


class Lab1Tests(unittest.TestCase):
    def assert_eigenvalues(self, values, expected, tolerance):
        error = float("inf")
        for order in itertools.permutations(expected):
            difference = max(abs(a - b) for a, b in zip(values, order))
            error = min(error, difference)
        self.assertLessEqual(error, tolerance)

    def assert_trace(self, trace, k):
        self.assertLessEqual(len(trace), 3)
        self.assertEqual(len({step[0] for step in trace}), len(trace))
        if trace:
            self.assertEqual(trace[-1][0], k)

    def check_lu(self, path):
        A, b = lu.read_system(path)
        L, U, P, p, trace = lu.lu_decomposition(A)
        Pb, z, x = lu.solve_lu(L, U, P, b)
        X = lu.inverse_from_lu(L, U, P)
        np.testing.assert_allclose(np.matmul(P, A), np.matmul(L, U), atol=1e-12)
        np.testing.assert_allclose(np.matmul(L, z), Pb, atol=1e-12)
        np.testing.assert_allclose(np.matmul(U, x), z, atol=1e-12)
        np.testing.assert_allclose(x, np.linalg.solve(A, b), atol=1e-12)
        np.testing.assert_allclose(X, np.linalg.inv(A), atol=1e-12)
        det = lu.determinant_from_lu(U, p)
        self.assertAlmostEqual(det, np.linalg.det(A))
        self.assert_trace(trace, len(A))
        return x, p, det

    def test_lu_book(self):
        x, p, det = self.check_lu(fixture("lu_book.txt"))
        np.testing.assert_allclose(x, [1, 1, 1], atol=1e-12)
        self.assertAlmostEqual(det, 946)

    def test_lu_pivots_and_single(self):
        x, p, det = self.check_lu(ROOT / "data" / "task_1_1.txt")
        self.assertEqual(p, 2)
        x, p, det = self.check_lu(fixture("lu_odd_swap.txt"))
        self.assertEqual(p, 1)
        self.assertAlmostEqual(det, -2)
        self.check_lu(fixture("lu_single.txt"))

    def test_lu_singular(self):
        A, b = lu.read_system(fixture("lu_singular.txt"))
        with self.assertRaises(ValueError):
            lu.lu_decomposition(A)

    def test_sweep(self):
        for name in ("sweep_book.txt", "sweep_single.txt", "sweep_not_dominant.txt"):
            with self.subTest(name=name):
                a, b, c, d = sweep.read_system(fixture(name))
                P, Q = sweep.sweep_coefficients(a, b, c, d)
                x = sweep.back_substitution(P, Q)
                A = np.diag(b) + np.diag(a[1:], -1) + np.diag(c[:-1], 1)
                np.testing.assert_allclose(x, np.linalg.solve(A, d), atol=1e-12)
                self.assertEqual(P[-1], 0)
                self.assertLess(sweep.residual(a, b, c, d, x), 1e-12)
                if name == "sweep_book.txt":
                    np.testing.assert_allclose(x, [1, 1, 1, 1], atol=1e-12)

    def test_sweep_zero_denominator(self):
        a, b, c, d = sweep.read_system(fixture("sweep_zero.txt"))
        with self.assertRaises(ValueError):
            sweep.sweep_coefficients(a, b, c, d)

    def check_iterations(self, path):
        A, b, eps, x_ref = iterations.read_problem(path)
        alpha, beta = iterations.iteration_form(A, b)
        B, C = iterations.split_matrix(alpha)
        np.testing.assert_allclose(np.add(B, C), alpha)
        expected = np.linalg.solve(A, b)
        np.testing.assert_allclose(x_ref, expected, atol=1e-12)
        results = []
        for method in (iterations.jacobi_iteration, iterations.seidel_iteration):
            x, k, trace = iterations.solve_iterations(alpha, beta, eps, method)
            self.assertLessEqual(matrices.vector_difference(x, expected), eps)
            self.assertLessEqual(trace[-1][3], eps)
            self.assert_trace(trace, k)
            results.append((x, k, trace))
        return results

    def test_iterations_book(self):
        jacobi, seidel = self.check_iterations(fixture("iteration_book.txt"))
        self.assertEqual(jacobi[1], 4)
        self.assertEqual(seidel[1], 3)
        np.testing.assert_allclose(jacobi[2][0][1], [0.93, 0.92, 0.9])
        np.testing.assert_allclose(seidel[2][0][1], [0.93, 0.974, 1.0192])
        self.assertLess(matrices.vector_difference(seidel[2][1][1], [1, 1, 1]), 0.01)
        self.assertGreater(seidel[2][1][3], 0.01)

    def test_iterations_current_and_single(self):
        self.check_iterations(ROOT / "data" / "task_1_3.txt")
        self.check_iterations(fixture("iteration_single.txt"))

    def test_iterations_without_dominance(self):
        self.check_iterations(fixture("iteration_not_dominant.txt"))

    def test_iteration_limit(self):
        A, b, eps, x_ref = iterations.read_problem(fixture("iteration_divergent.txt"))
        alpha, beta = iterations.iteration_form(A, b)
        for method in (iterations.jacobi_iteration, iterations.seidel_iteration):
            with self.assertRaises(ValueError):
                iterations.solve_iterations(alpha, beta, eps, method, max_iter=5)

    def check_rotation(self, path):
        A0, eps = rotations.read_problem(path)
        A, V, k, trace = rotations.rotation_method(A0, eps)
        self.assertLess(matrices.off_diagonal_norm(A), eps)
        values = [A[i][i] for i in range(len(A))]
        self.assert_eigenvalues(values, np.linalg.eigvalsh(A0), 2 * eps)
        V = np.array(V)
        residual = np.matmul(A0, V) - np.matmul(V, np.diag(values))
        self.assertLessEqual(np.max(np.abs(residual)), 2 * eps + 1e-12)
        np.testing.assert_allclose(V.T @ V, np.eye(len(A)), atol=1e-12)
        self.assert_trace(trace, k)
        return k

    def test_rotations(self):
        self.check_rotation(ROOT / "data" / "task_1_4.txt")
        self.assertEqual(self.check_rotation(fixture("rotation_book.txt")), 4)
        self.assertEqual(self.check_rotation(fixture("rotation_single.txt")), 0)
        self.assertEqual(self.check_rotation(fixture("rotation_diagonal.txt")), 0)
        self.assertEqual(self.check_rotation(fixture("rotation_equal_diagonal.txt")), 1)

    def test_rotation_book_stopping(self):
        A0, eps = rotations.read_problem(fixture("rotation_book.txt"))
        A, V, k, trace = rotations.rotation_method(A0, eps)
        A2 = trace[1][6]
        i, j = rotations.largest_off_diagonal(A2)
        U, phi = rotations.jacobi_rotation(A2, i, j)
        A3 = np.transpose(U) @ A2 @ U
        self.assertAlmostEqual(matrices.off_diagonal_norm(A3), 0.3163623161)
        self.assertGreater(matrices.off_diagonal_norm(A3), eps)

    def test_rotation_asymmetric(self):
        with self.assertRaises(ValueError):
            rotations.read_problem(fixture("rotation_asymmetric.txt"))

    def check_qr(self, path):
        A, eps, example = qr.read_problem(path)
        result = None
        for matrix in (A, example):
            Q, R = qr.qr_decomposition(matrix)
            np.testing.assert_allclose(np.matmul(Q, R), matrix, atol=1e-12)
            np.testing.assert_allclose(np.transpose(Q) @ Q, np.eye(len(matrix)), atol=1e-12)
            np.testing.assert_allclose(np.tril(R, -1), 0, atol=1e-12)
            values, k, trace = qr.qr_eigenvalues(matrix, eps)
            self.assert_eigenvalues(values, np.linalg.eigvals(matrix), 2 * eps + 1e-12)
            self.assert_trace(trace, k)
            if result is None:
                result = values, k, trace
        return result

    def test_qr_book_and_complex_pair(self):
        self.check_qr(fixture("qr_book.txt"))
        A, eps, example = qr.read_problem(fixture("qr_single.txt"))
        values, k, trace = qr.qr_eigenvalues(example, eps)
        self.assert_eigenvalues(values, [1j, -1j], 1e-12)

    def test_qr_current_regression(self):
        A, eps, example = qr.read_problem(ROOT / "data" / "task_1_5.txt")
        self.assertEqual(A[2][1], 0)
        self.assertNotEqual(A[2][0], 0)
        values, checks = qr.isolated_eigenvalues(A, A, eps)
        self.assertIsNone(values)
        values, k, trace = self.check_qr(ROOT / "data" / "task_1_5.txt")
        self.assertGreater(k, 1)
        self.assertFalse(any(abs(value - 2) <= eps for value in values))

    def test_qr_single_diagonal_and_separation(self):
        self.check_qr(fixture("qr_single.txt"))
        values, k, trace = self.check_qr(fixture("qr_diagonal.txt"))
        self.assertEqual(k, 0)
        A, eps, example = qr.read_problem(fixture("qr_not_isolated.txt"))
        self.assertNotEqual(qr.block_eigenvalues(A, 0)[0].imag, 0)
        values, checks = qr.isolated_eigenvalues(A, A, eps)
        self.assertIsNone(values)

    def test_qr_limit(self):
        A, eps, example = qr.read_problem(fixture("qr_stagnation.txt"))
        with self.assertRaises(ValueError):
            qr.qr_eigenvalues(A, eps, max_iter=5)

    def test_larger_matrices(self):
        self.check_lu(fixture("lu_larger.txt"))
        self.check_iterations(fixture("iteration_larger.txt"))
        self.check_rotation(fixture("rotation_larger.txt"))
        self.check_qr(fixture("qr_larger.txt"))
        a, b, c, d = sweep.read_system(fixture("sweep_larger.txt"))
        P, Q = sweep.sweep_coefficients(a, b, c, d)
        x = sweep.back_substitution(P, Q)
        np.testing.assert_allclose(x, np.ones(len(x)), atol=1e-12)

    def test_qr_two_complex_pairs(self):
        self.check_qr(fixture("qr_two_pairs.txt"))

    def test_invalid_inputs(self):
        for name in ("invalid_size.txt", "invalid_truncated.txt", "invalid_extra.txt", "invalid_finite.txt"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                lu.read_system(fixture(name))
        with self.assertRaises(ValueError):
            rotations.read_problem(fixture("invalid_eps.txt"))
        with self.assertRaises(ValueError):
            input_output.read_eps("-1")
        with self.assertRaises(ValueError):
            input_output.read_eps("nan")

    def test_source_style(self):
        for path in ROOT.glob("*.py"):
            source = path.read_text(encoding="utf-8-sig")
            tree = ast.parse(source)
            self.assertIsNone(ast.get_docstring(tree))
            tokens = tokenize.generate_tokens(io.StringIO(source).readline)
            self.assertFalse(any(token.type == tokenize.COMMENT for token in tokens))
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    self.assertIsNone(ast.get_docstring(node))
                    self.assertIsNone(node.returns)
                    for arg in node.args.args:
                        self.assertIsNone(arg.annotation)
                    self.assertFalse(any(isinstance(child, ast.FunctionDef) for child in node.body))

    def test_cli(self):
        programs = (
            "task_1_1_lu_decomposition.py",
            "task_1_2_sweep_method.py",
            "task_1_3_jacobi_and_seidel_methods.py",
            "task_1_4_jacobi_rotation_method.py",
            "task_1_5_qr_eigenvalues.py",
        )
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        for i, program in enumerate(programs, 1):
            with self.subTest(program=program):
                path = ROOT / "data" / f"task_1_{i}.txt"
                result = subprocess.run(
                    [sys.executable, str(ROOT / program), str(path)],
                    capture_output=True, encoding="utf-8", env=env, timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertGreater(len(result.stdout), 200)
                self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
