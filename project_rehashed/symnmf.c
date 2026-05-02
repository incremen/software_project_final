#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "symnmf.h"
#include "matrix_utils.h"

#define BETA 0.5
#define EPS 1e-4
#define MAX_ITER 300
#define LINE_BUF 4096

/* ---------- top-level entry points ---------- */

static void fail_and_exit(void);
static double **read_data_file(const char *path, int *num_points, int *dim);
static double **dispatch_goal(const char *goal, double **data, int n, int d);

/* CLI: ./symnmf <sym|ddg|norm> <file.txt>. Prints the requested matrix. */
int main(int argc, char *argv[]) {
    int num_points, dim;
    double **data, **result;

    if (argc != 3) fail_and_exit();
    data = read_data_file(argv[2], &num_points, &dim);
    result = dispatch_goal(argv[1], data, num_points, dim);

    print_matrix(result, num_points, num_points);
    free_matrix(result, num_points);
    free_matrix(data, num_points);
    return 0;
}

/* Route goal name to the matching compute_* routine. */
static double **dispatch_goal(const char *goal, double **data, int n, int d) {
    if (strcmp(goal, "sym") == 0)  return compute_sym(data, n, d);
    if (strcmp(goal, "ddg") == 0)  return compute_ddg(data, n, d);
    if (strcmp(goal, "norm") == 0) return compute_norm(data, n, d);
    free_matrix(data, n);
    fail_and_exit();
    return NULL; /* unreachable */
}

/* ---------- symNMF math ---------- */

/* Squared Euclidean distance between two dim-length vectors. */
static double squared_distance(double *a, double *b, int dim) {
    int axis;
    double total = 0.0;
    for (axis = 0; axis < dim; axis++) {
        total += (a[axis] - b[axis]) * (a[axis] - b[axis]);
    }
    return total;
}

/* Similarity matrix A: a_ij = exp(-||x_i - x_j||^2 / 2), zero on diagonal. */
double **compute_sym(double **data, int num_points, int dim) {
    int i, j;
    double **similarity = allocate_matrix(num_points, num_points);
    for (i = 0; i < num_points; i++) {
        for (j = 0; j < num_points; j++) {
            if (i != j) {
                similarity[i][j] =
                    exp(-squared_distance(data[i], data[j], dim) / 2.0);
            }
        }
    }
    return similarity;
}

/* Diagonal degree matrix D with d_i = sum_j A_ij on the diagonal. */
double **compute_ddg(double **data, int num_points, int dim) {
    int i, j;
    double **similarity = compute_sym(data, num_points, dim);
    double **degree = allocate_matrix(num_points, num_points);
    for (i = 0; i < num_points; i++) {
        for (j = 0; j < num_points; j++) {
            degree[i][i] += similarity[i][j];
        }
    }
    free_matrix(similarity, num_points);
    return degree;
}

/* Normalized similarity W = D^{-1/2} A D^{-1/2}. */
double **compute_norm(double **data, int num_points, int dim) {
    int i, j;
    double **similarity = compute_sym(data, num_points, dim);
    double **normalized = allocate_matrix(num_points, num_points);
    double *degrees = (double *)calloc(num_points, sizeof(double));
    if (degrees == NULL) {
        free_matrix(similarity, num_points);
        free_matrix(normalized, num_points);
        fail_and_exit();
    }
    for (i = 0; i < num_points; i++)
        for (j = 0; j < num_points; j++) degrees[i] += similarity[i][j];
    for (i = 0; i < num_points; i++) {
        for (j = 0; j < num_points; j++) {
            normalized[i][j] =
                similarity[i][j] / (sqrt(degrees[i]) * sqrt(degrees[j]));
        }
    }
    free_matrix(similarity, num_points);
    free(degrees);
    return normalized;
}

/* Multiplicative update for one iteration of the symNMF rule. */
static double update_assoc_inplace(double **assoc, double **wh,
                                   double **hhth, int n, int k) {
    int i, j;
    double previous, change = 0.0;
    for (i = 0; i < n; i++) {
        for (j = 0; j < k; j++) {
            previous = assoc[i][j];
            assoc[i][j] = previous *
                (1 - BETA + BETA * wh[i][j] / hhth[i][j]);
            change += (assoc[i][j] - previous) * (assoc[i][j] - previous);
        }
    }
    return change;
}

/* Iteratively refine H so that ||W - H H^T||_F^2 decreases. */
double **optimize_symnmf(double **norm_sim, double **assoc,
                         int num_points, int num_clusters) {
    int iteration;
    double **wh, **h_t, **hht, **hhth;
    double change;
    for (iteration = 0; iteration < MAX_ITER; iteration++) {
        wh = multiply_matrices(norm_sim, assoc,
                               num_points, num_points, num_clusters);
        h_t = transpose_matrix(assoc, num_points, num_clusters);
        hht = multiply_matrices(assoc, h_t,
                                num_points, num_clusters, num_points);
        free_matrix(h_t, num_clusters);
        hhth = multiply_matrices(hht, assoc,
                                 num_points, num_points, num_clusters);
        free_matrix(hht, num_points);
        change = update_assoc_inplace(assoc, wh, hhth,
                                      num_points, num_clusters);
        free_matrix(wh, num_points);
        free_matrix(hhth, num_points);
        if (change < EPS) break;
    }
    return assoc;
}

/* ---------- I/O helpers ---------- */

/* Count rows and infer dimensionality from the first non-blank line. */
static void scan_dimensions(FILE *fp, int *num_points, int *dim) {
    char line[LINE_BUF];
    char *cursor;
    int rows = 0, cols = 1;
    while (fgets(line, LINE_BUF, fp) != NULL) {
        if (line[0] == '\n' || line[0] == '\r' || line[0] == '\0') continue;
        if (rows == 0) {
            for (cursor = line; *cursor; cursor++)
                if (*cursor == ',') cols++;
        }
        rows++;
    }
    *num_points = rows;
    *dim = cols;
}

/* Parse the CSV file into a freshly allocated (num_points x dim) matrix. */
static double **read_data_file(const char *path, int *num_points, int *dim) {
    FILE *fp = fopen(path, "r");
    char line[LINE_BUF];
    char *token;
    int i = 0, j;
    double **data;
    if (fp == NULL) fail_and_exit();
    scan_dimensions(fp, num_points, dim);
    data = allocate_matrix(*num_points, *dim);
    rewind(fp);
    while (i < *num_points && fgets(line, LINE_BUF, fp) != NULL) {
        if (line[0] == '\n' || line[0] == '\r' || line[0] == '\0') continue;
        token = strtok(line, ",\n\r");
        for (j = 0; j < *dim && token != NULL; j++) {
            data[i][j] = atof(token);
            token = strtok(NULL, ",\n\r");
        }
        i++;
    }
    fclose(fp);
    return data;
}

/* Emit canonical failure message and terminate. */
static void fail_and_exit(void) {
    printf("An Error Has Occurred\n");
    exit(1);
}
