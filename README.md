# Numerical Methods — Laboratory Work 1

Five standalone programs implement the methods listed in the laboratory assignment. They use only the Python standard library and read all problem data from text files.

## Programs

| Assignment | Program | Method |
| --- | --- | --- |
| 1.1 | `task_1_1_lu_decomposition.py` | LU decomposition with partial pivoting |
| 1.2 | `task_1_2_thomas_algorithm.py` | Thomas algorithm for tridiagonal systems |
| 1.3 | `task_1_3_iterative_methods.py` | Jacobi and Gauss–Seidel iterations |
| 1.4 | `task_1_4_jacobi_rotations.py` | Jacobi rotations for symmetric eigenproblems |
| 1.5 | `task_1_5_qr_eigenvalues.py` | QR decomposition and QR eigenvalue iteration |

## Run

From this directory, run a program with its sample input file:

```text
python task_1_1_lu_decomposition.py data/task_1_1.txt
python task_1_2_thomas_algorithm.py data/task_1_2.txt
python task_1_3_iterative_methods.py data/task_1_3.txt
python task_1_4_jacobi_rotations.py data/task_1_4.txt
python task_1_5_qr_eigenvalues.py data/task_1_5.txt
```

Each input file starts with the dimension where applicable, followed by matrix rows or diagonal vectors. Iterative methods also read the tolerance and a reference solution. The QR input contains a second matrix used to demonstrate a complex-conjugate eigenvalue pair.

## Input layouts

`task_1_1`: `n`, `n` rows of `A`, then `b`.

`task_1_2`: `n`, lower diagonal, main diagonal, upper diagonal, right-hand side. Each diagonal contains `n` values; the unused first lower and last upper entries are zero.

`task_1_3`: `n`, `n` rows of `A`, `b`, tolerance, reference solution.

`task_1_4`: `n`, `n` rows of a real symmetric matrix, tolerance.

`task_1_5`: `n`, `n` rows of `A`, tolerance, dimension `m` of the demonstration matrix, then `m` rows of that matrix.

The supplied files in `data` are small examples. Replace their values or pass another file path to solve a different problem; matrix dimensions are read from the data.
