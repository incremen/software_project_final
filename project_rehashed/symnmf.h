#ifndef SYMNMF_H
#define SYMNMF_H

double **compute_sym(double **data, int num_points, int dim);
double **compute_ddg(double **data, int num_points, int dim);
double **compute_norm(double **data, int num_points, int dim);
double **optimize_symnmf(double **norm_sim, double **assoc,
                         int num_points, int num_clusters);

#endif
