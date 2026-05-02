#include <stdio.h>
#include <stdlib.h>
#include "matrix_utils.h"

/* Emit canonical failure message and terminate. */
static void fail_and_exit(void) {
    printf("An Error Has Occurred\n");
    exit(1);
}

/* Zero-initialized rows x cols matrix; aborts on allocation failure. */
double **allocate_matrix(int rows, int cols) {
    int row_index;
    double **matrix = (double **)calloc(rows, sizeof(double *));
    if (matrix == NULL) fail_and_exit();
    for (row_index = 0; row_index < rows; row_index++) {
        matrix[row_index] = (double *)calloc(cols, sizeof(double));
        if (matrix[row_index] == NULL) fail_and_exit();
    }
    return matrix;
}

/* Release a matrix previously returned by allocate_matrix. */
void free_matrix(double **matrix, int rows) {
    int row_index;
    if (matrix == NULL) return;
    for (row_index = 0; row_index < rows; row_index++) {
        free(matrix[row_index]);
    }
    free(matrix);
}

/* Write matrix as CSV with 4-decimal precision, one row per line. */
void print_matrix(double **matrix, int rows, int cols) {
    int row_index, col_index;
    for (row_index = 0; row_index < rows; row_index++) {
        for (col_index = 0; col_index < cols; col_index++) {
            if (col_index > 0) printf(",");
            printf("%.4f", matrix[row_index][col_index]);
        }
        printf("\n");
    }
}

/* Standard (rows x inner) * (inner x cols) matrix product. */
double **multiply_matrices(double **left, double **right,
                           int rows, int inner, int cols) {
    int row_index, col_index, k_index;
    double **product = allocate_matrix(rows, cols);
    for (row_index = 0; row_index < rows; row_index++) {
        for (col_index = 0; col_index < cols; col_index++) {
            for (k_index = 0; k_index < inner; k_index++) {
                product[row_index][col_index] +=
                    left[row_index][k_index] * right[k_index][col_index];
            }
        }
    }
    return product;
}

/* Transpose of a (rows x cols) matrix into a freshly allocated (cols x rows). */
double **transpose_matrix(double **matrix, int rows, int cols) {
    int row_index, col_index;
    double **transposed = allocate_matrix(cols, rows);
    for (row_index = 0; row_index < rows; row_index++) {
        for (col_index = 0; col_index < cols; col_index++) {
            transposed[col_index][row_index] = matrix[row_index][col_index];
        }
    }
    return transposed;
}
