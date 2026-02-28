#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "symnmf.h"

#define BETA 0.5
#define EPS 1e-4
#define MAX_ITER 300
#define LINE_BUF 4096

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
static void print_matrix(double **mat, int n, int m) {
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
static double **mat_mult(double **A, double **B, int n, int m, int p) {
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
static double **transpose(double **A, int n, int m) {
    int i, j;
    double **T = alloc_matrix(m, n);
    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            T[j][i] = A[i][j];
        }
    }
    return T;
}

/* Squared Euclidean distance between d-dimensional vectors */
static double sq_euclidean(double *a, double *b, int d) {
    int i;
    double sum = 0.0;
    for (i = 0; i < d; i++) {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sum;
}

/* Compute similarity matrix A from data X (n x d) */
double **compute_sym(double **X, int n, int d) {
    int i, j;
    double **A = alloc_matrix(n, n);
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (i != j) {
                A[i][j] = exp(-sq_euclidean(X[i], X[j], d) / 2.0);
            }
        }
    }
    return A;
}

/* Compute diagonal degree matrix D from data X (n x d) */
double **compute_ddg(double **X, int n, int d) {
    int i, j;
    double **A = compute_sym(X, n, d);
    double **D = alloc_matrix(n, n);
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            D[i][i] += A[i][j];
        }
    }
    free_matrix(A, n);
    return D;
}

/* Compute normalized similarity W = D^{-1/2} * A * D^{-1/2} */
double **compute_norm(double **X, int n, int d) {
    int i, j;
    double **A = compute_sym(X, n, d);
    double **W = alloc_matrix(n, n);
    double *deg = (double *)calloc(n, sizeof(double));
    if (deg == NULL) {
        free_matrix(A, n);
        free_matrix(W, n);
        error_exit();
    }
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            deg[i] += A[i][j];
        }
    }
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            W[i][j] = A[i][j] / (sqrt(deg[i]) * sqrt(deg[j]));
        }
    }
    free_matrix(A, n);
    free(deg);
    return W;
}

/* Optimize H using the symNMF iterative update rule */
double **optimize_symnmf(double **W, double **H, int n, int k) {
    int iter, i, j;
    double **WH, **HT, **HHT, **HHTH;
    double diff, old_val;
    for (iter = 0; iter < MAX_ITER; iter++) {
        WH = mat_mult(W, H, n, n, k);
        HT = transpose(H, n, k);
        HHT = mat_mult(H, HT, n, k, n);
        free_matrix(HT, k);
        HHTH = mat_mult(HHT, H, n, n, k);
        free_matrix(HHT, n);
        diff = 0.0;
        for (i = 0; i < n; i++) {
            for (j = 0; j < k; j++) {
                old_val = H[i][j];
                H[i][j] = old_val * (1 - BETA +
                    BETA * WH[i][j] / HHTH[i][j]);
                diff += (H[i][j] - old_val) * (H[i][j] - old_val);
            }
        }
        free_matrix(WH, n);
        free_matrix(HHTH, n);
        if (diff < EPS) {
            break;
        }
    }
    return H;
}

/* Read data points from file, set n (rows) and d (dimension) */
static double **read_data(const char *filename, int *n, int *d) {
    FILE *fp;
    char line[LINE_BUF];
    char *token, *p;
    int i, j, count = 0, dim = 1;
    double **X;
    fp = fopen(filename, "r");
    if (fp == NULL) error_exit();
    while (fgets(line, LINE_BUF, fp) != NULL) {
        if (line[0] == '\n' || line[0] == '\r' || line[0] == '\0')
            continue;
        if (count == 0) {
            for (p = line; *p; p++)
                if (*p == ',') dim++;
        }
        count++;
    }
    *n = count;
    *d = dim;
    X = alloc_matrix(count, dim);
    rewind(fp);
    i = 0;
    while (i < count && fgets(line, LINE_BUF, fp) != NULL) {
        if (line[0] == '\n' || line[0] == '\r' || line[0] == '\0')
            continue;
        token = strtok(line, ",\n\r");
        for (j = 0; j < dim && token != NULL; j++) {
            X[i][j] = atof(token);
            token = strtok(NULL, ",\n\r");
        }
        i++;
    }
    fclose(fp);
    return X;
}

/* Main: C interface for sym, ddg, norm */
int main(int argc, char *argv[]) {
    char *goal, *filename;
    int n, d;
    double **X, **result = NULL;
    if (argc != 3) error_exit();
    goal = argv[1];
    filename = argv[2];
    X = read_data(filename, &n, &d);
    if (strcmp(goal, "sym") == 0) {
        result = compute_sym(X, n, d);
    } else if (strcmp(goal, "ddg") == 0) {
        result = compute_ddg(X, n, d);
    } else if (strcmp(goal, "norm") == 0) {
        result = compute_norm(X, n, d);
    } else {
        free_matrix(X, n);
        error_exit();
    }
    print_matrix(result, n, n);
    free_matrix(result, n);
    free_matrix(X, n);
    return 0;
}
