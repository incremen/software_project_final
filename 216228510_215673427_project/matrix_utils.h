#ifndef MATRIX_UTILS_H
#define MATRIX_UTILS_H

double **alloc_matrix(int rows, int cols);
void free_matrix(double **mat, int rows);
void print_matrix(double **mat, int n, int m);
double **mat_mult(double **A, double **B, int n, int m, int p);
double **transpose(double **A, int n, int m);

#endif
