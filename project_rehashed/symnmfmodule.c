#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "symnmf.h"
#include "matrix_utils.h"

/* ---------- Python bindings (listed top-down) ---------- */

static PyObject *sym_binding(PyObject *self, PyObject *args);
static PyObject *ddg_binding(PyObject *self, PyObject *args);
static PyObject *norm_binding(PyObject *self, PyObject *args);
static PyObject *symnmf_binding(PyObject *self, PyObject *args);

static double **py_list_to_matrix(PyObject *py_matrix, int rows, int cols);
static PyObject *matrix_to_py_list(double **matrix, int rows, int cols);

static PyMethodDef symnmf_methods[] = {
    {"sym",    sym_binding,    METH_VARARGS, "Compute the similarity matrix."},
    {"ddg",    ddg_binding,    METH_VARARGS, "Compute the diagonal degree matrix."},
    {"norm",   norm_binding,   METH_VARARGS, "Compute the normalized similarity matrix."},
    {"symnmf", symnmf_binding, METH_VARARGS, "Run the full symNMF optimization."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef symnmf_module_def = {
    PyModuleDef_HEAD_INIT,
    "symnmf",
    "SymNMF C extension.",
    -1,
    symnmf_methods
};

PyMODINIT_FUNC PyInit_symnmf(void) {
    return PyModule_Create(&symnmf_module_def);
}

/* ---------- single-argument (datapoints) wrappers ---------- */

/* Build A from a Python list-of-lists of points. */
static PyObject *sym_binding(PyObject *self, PyObject *args) {
    PyObject *py_points;
    int num_points, dim;
    double **points, **similarity;
    PyObject *result;
    (void)self;

    if (!PyArg_ParseTuple(args, "O", &py_points)) return NULL;
    num_points = (int)PyObject_Length(py_points);
    dim = (int)PyObject_Length(PyList_GetItem(py_points, 0));

    points = py_list_to_matrix(py_points, num_points, dim);
    similarity = compute_sym(points, num_points, dim);
    result = matrix_to_py_list(similarity, num_points, num_points);

    free_matrix(points, num_points);
    free_matrix(similarity, num_points);
    return result;
}

/* Build D from a Python list-of-lists of points. */
static PyObject *ddg_binding(PyObject *self, PyObject *args) {
    PyObject *py_points;
    int num_points, dim;
    double **points, **degree;
    PyObject *result;
    (void)self;

    if (!PyArg_ParseTuple(args, "O", &py_points)) return NULL;
    num_points = (int)PyObject_Length(py_points);
    dim = (int)PyObject_Length(PyList_GetItem(py_points, 0));

    points = py_list_to_matrix(py_points, num_points, dim);
    degree = compute_ddg(points, num_points, dim);
    result = matrix_to_py_list(degree, num_points, num_points);

    free_matrix(points, num_points);
    free_matrix(degree, num_points);
    return result;
}

/* Build W from a Python list-of-lists of points. */
static PyObject *norm_binding(PyObject *self, PyObject *args) {
    PyObject *py_points;
    int num_points, dim;
    double **points, **normalized;
    PyObject *result;
    (void)self;

    if (!PyArg_ParseTuple(args, "O", &py_points)) return NULL;
    num_points = (int)PyObject_Length(py_points);
    dim = (int)PyObject_Length(PyList_GetItem(py_points, 0));

    points = py_list_to_matrix(py_points, num_points, dim);
    normalized = compute_norm(points, num_points, dim);
    result = matrix_to_py_list(normalized, num_points, num_points);

    free_matrix(points, num_points);
    free_matrix(normalized, num_points);
    return result;
}

/* Run the full symNMF optimization given W and an initial H. */
static PyObject *symnmf_binding(PyObject *self, PyObject *args) {
    PyObject *py_norm, *py_init_assoc, *result;
    int num_points, num_clusters;
    double **norm_sim, **assoc, **final_assoc;
    (void)self;

    if (!PyArg_ParseTuple(args, "OO", &py_norm, &py_init_assoc)) return NULL;
    num_points = (int)PyObject_Length(py_norm);
    num_clusters = (int)PyObject_Length(PyList_GetItem(py_init_assoc, 0));

    assoc = py_list_to_matrix(py_init_assoc, num_points, num_clusters);
    norm_sim = py_list_to_matrix(py_norm, num_points, num_points);
    final_assoc = optimize_symnmf(norm_sim, assoc, num_points, num_clusters);
    result = matrix_to_py_list(final_assoc, num_points, num_clusters);

    free_matrix(assoc, num_points);
    free_matrix(norm_sim, num_points);
    return result;
}

/* ---------- Python <-> C matrix conversions ---------- */

/* Copy a Python list-of-lists into a freshly allocated C matrix. */
static double **py_list_to_matrix(PyObject *py_matrix, int rows, int cols) {
    int i, j;
    double **matrix = allocate_matrix(rows, cols);
    for (i = 0; i < rows; i++) {
        PyObject *row = PyList_GetItem(py_matrix, i);
        for (j = 0; j < cols; j++) {
            matrix[i][j] = PyFloat_AsDouble(PyList_GetItem(row, j));
        }
    }
    return matrix;
}

/* Copy a C matrix into a new Python list-of-lists of floats. */
static PyObject *matrix_to_py_list(double **matrix, int rows, int cols) {
    int i, j;
    PyObject *py_matrix = PyList_New(rows);
    for (i = 0; i < rows; i++) {
        PyObject *py_row = PyList_New(cols);
        for (j = 0; j < cols; j++) {
            PyList_SetItem(py_row, j, PyFloat_FromDouble(matrix[i][j]));
        }
        PyList_SetItem(py_matrix, i, py_row);
    }
    return py_matrix;
}
