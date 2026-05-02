#ifndef MATRIX_UTILS_H
#define MATRIX_UTILS_H

double **allocate_matrix(int rows, int cols);
void free_matrix(double **matrix, int rows);
void print_matrix(double **matrix, int rows, int cols);
double **multiply_matrices(double **left, double **right,
                           int rows, int inner, int cols);
double **transpose_matrix(double **matrix, int rows, int cols);

#endif
