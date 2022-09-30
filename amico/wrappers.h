#include "nnls.h"
#include "spams.h"

// IN
// A = (m, n)   Matrix stored as 1D contiguous array (column-major order)
// y = (m,)     Vector stored as 1D contiguous array

// OUT
// x = (n,)     Solution vector
void nnls
(
    const double *A,
    const double *y,
    const int m,
    const int n,
    double *x,
    double *rnorm
)
{
    // Make a copy of A and y
    double *A_copy = (double *) malloc(sizeof(double) * m * n);
    double *y_copy = (double *) malloc(sizeof(double) * m * n);
    std::copy(A, A + m * n, A_copy);
    std::copy(y, y + m, y_copy);

    // Instantiate other arrays of working space
    double *wp = NULL;
    double *zzp = NULL;
    int *indexp = NULL;

    // Call nnls algorithm
    _nnls(A_copy, m, n, y_copy, x, rnorm, wp, zzp, indexp);

    // Free memory
    free(A_copy);
    free(y_copy);
}

// IN
// A = (m, n)   Matrix stored as 1D contiguous array (column-major order)
// y = (m,)     Vector stored as 1D contiguous array

// OUT
// x = (n,)     Solution vector
void lasso
(
    double *A,
    double *y,
    const int m,
    const int n,
    const double lambda1,
    const double lambda2,
    double *x
)
{
    Matrix<double> *_A = new Matrix<double>(A, m, n);
    Matrix<double> *_y = new Matrix<double>(y, m, 1);

    Matrix<double> **path = 0;
    Matrix<double> *path_data;
    path = &path_data;

    const bool return_reg_path = false;
    const int L = -1;
    const constraint_type mode = PENALTY;
    const bool pos = true;
    const bool ols = false;
    const int n_threads = 1;
    const int max_length_path = -1;
    const bool verbose = false;
    const bool cholevsky = false;

    SpMatrix<double> *_x = (SpMatrix<double> *)_lassoD(_y, _A, path, return_reg_path, L, lambda1, lambda2, mode, pos, ols, n_threads, max_length_path, verbose, cholevsky);

    // Copy x in non-sparse format
    Matrix<double> _x_dense;
    _x->toFull(_x_dense);

    std::copy(_x_dense.rawX(), _x_dense.rawX() + _x_dense.m() * _x_dense.n(), x);
}
