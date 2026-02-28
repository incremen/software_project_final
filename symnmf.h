#ifndef SYMNMF_H
#define SYMNMF_H

double **compute_sym(double **X, int n, int d);
double **compute_ddg(double **X, int n, int d);
double **compute_norm(double **X, int n, int d);
double **optimize_symnmf(double **W, double **H, int n, int k);

#endif
