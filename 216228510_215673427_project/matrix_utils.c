#include <stdio.h>
#include <stdlib.h>
#include "matrix_utils.h"

/* Print error message and exit */
static void error_exit(void) {
    printf("An Error Has Occurred\n");
    exit(1);
}

/* Allocate an rows x cols matrix initialized to zero */
double **alloc_matrix(int rows, int cols) {
    int i;
    double **mat = (double **)calloc(rows, sizeof(double *));
    if (mat == NULL) error_exit();
    for (i = 0; i < rows; i++) {
        mat[i] = (double *)calloc(cols, sizeof(double));
        if (mat[i] == NULL) error_exit();
    }
    return mat;
}

/* Free a matrix with the given number of rows */
void free_matrix(double **mat, int rows) {
    int i;
    if (mat == NULL) return;
    for (i = 0; i < rows; i++) {
        free(mat[i]);
    }
    free(mat);
}

/* Print n x m matrix, comma-separated, 4 decimal places */
void print_matrix(double **mat, int n, int m) {
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            if (j > 0) printf(",");
            printf("%.4f", mat[i][j]);
        }
        printf("\n");
    }
}

/* Multiply A (n x m) by B (m x p), return C (n x p) */
double **mat_mult(double **A, double **B, int n, int m, int p) {
    int i, j, l;
    double **C = alloc_matrix(n, p);
    for (i = 0; i < n; i++) {
        for (j = 0; j < p; j++) {
            for (l = 0; l < m; l++) {
                C[i][j] += A[i][l] * B[l][j];
            }
        }
    }
    return C;
}

/* Transpose A (n x m) -> result (m x n) */
double **transpose(double **A, int n, int m) {
    int i, j;
    double **T = alloc_matrix(m, n);
    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            T[j][i] = A[i][j];
        }
    }
    return T;
}
