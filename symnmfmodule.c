#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "symnmf.h" 
#include "matrix_utils.h"

// Converts a Python List of Lists into our contiguous C double** matrix
static double** py_list_to_c_matrix(PyObject* py_matrix, int n, int d) {
    int i, j;
    double** c_matrix = alloc_matrix(n, d); 

    for (i = 0; i < n; i++) {
        PyObject* row = PyList_GetItem(py_matrix, i);
        for (j = 0; j < d; j++) {
            PyObject* item = PyList_GetItem(row, j);
            c_matrix[i][j] = PyFloat_AsDouble(item);
        }
    }
    return c_matrix;
}

// Converts C double** matrix back into a Python List of Lists
static PyObject* c_matrix_to_py_list(double** c_matrix, int rows, int cols) {
    int i, j;
    PyObject* py_matrix = PyList_New(rows);

    for (i = 0; i < rows; i++) {
        PyObject* py_row = PyList_New(cols);
        for (j = 0; j < cols; j++) {
            PyList_SetItem(py_row, j, PyFloat_FromDouble(c_matrix[i][j]));
        }
        PyList_SetItem(py_matrix, i, py_row);
    }
    return py_matrix;
}

// Wrapper for the Similarity Matrix (sym)
static PyObject* sym_wrapper(PyObject* self, PyObject* args) {
    PyObject* py_datapoints;
    int n, d;

    // Parse the incoming Python arguments (expecting a list)
    if (!PyArg_ParseTuple(args, "O", &py_datapoints)) {
        return NULL; // PyArg_ParseTuple raises its own exception
    }

    // Get dimensions
    n = PyObject_Length(py_datapoints);
    d = PyObject_Length(PyList_GetItem(py_datapoints, 0));
    // printf("Received n: %d, d: %d\n", n, d); // Debug print
    // Convert to C
    double** X = py_list_to_c_matrix(py_datapoints, n, d);

    double** A = compute_sym(X, n, d); 

    // 3. Convert back to Python
    PyObject* py_result = c_matrix_to_py_list(A, n, n);

    // 4. CLEAN UP MEMORY! (Crucial step)
    free_matrix(X,n);
    free_matrix(A,n);

    return py_result;
}

// 2. Wrapper for the Diagonal Degree Matrix (ddg)
static PyObject* ddg_wrapper(PyObject* self, PyObject* args) {
    PyObject* py_datapoints;
    int n, d;

    // Parse the incoming Python arguments (expecting a list)
    if (!PyArg_ParseTuple(args, "O", &py_datapoints)) {
        return NULL; // PyArg_ParseTuple raises its own exception
    }

    // Get dimensions
    n = PyObject_Length(py_datapoints);
    d = PyObject_Length(PyList_GetItem(py_datapoints, 0));

    // Convert to C
    double** X = py_list_to_c_matrix(py_datapoints, n, d);

    double** A = compute_ddg(X, n, d); 

    // 3. Convert back to Python
    PyObject* py_result = c_matrix_to_py_list(A, n, n);

    free_matrix(X, n);
    free_matrix(A, n);

    return py_result;
}

// 3. Wrapper for the Normalized Similarity Matrix (norm)
static PyObject* norm_wrapper(PyObject* self, PyObject* args) {
    PyObject* py_datapoints;
    int n, d;

    // Parse the incoming Python arguments (expecting a list)
    if (!PyArg_ParseTuple(args, "O", &py_datapoints)) {
        return NULL; // PyArg_ParseTuple raises its own exception
    }

    // Get dimensions
    n = PyObject_Length(py_datapoints);
    d = PyObject_Length(PyList_GetItem(py_datapoints, 0));

    // Convert to C
    double** X = py_list_to_c_matrix(py_datapoints, n, d);

    double** A = compute_norm(X, n, d); 

    // 3. Convert back to Python
    PyObject* py_result = c_matrix_to_py_list(A, n, n);

    free_matrix(X, n);
    free_matrix(A, n);

    return py_result;
}

// 4. Wrapper for the full SymNMF algorithm
static PyObject* symnmf_wrapper(PyObject* self, PyObject* args) {
    PyObject *py_W, *py_H;
    int n, k;

    // Python passes W (n x n) first, then H (n x k)
    if (!PyArg_ParseTuple(args, "OO", &py_W, &py_H)) {
        return NULL; 
    }

    n = PyObject_Length(py_W);
    k = PyObject_Length(PyList_GetItem(py_H, 0));

        // Convert to C
    double** H = py_list_to_c_matrix(py_H, n, k);
    double** W = py_list_to_c_matrix(py_W, n, n);
    double** result = optimize_symnmf(W, H, n, k);
    PyObject* py_result = c_matrix_to_py_list(result, n, k);
    // Free memory
    free_matrix(H, n);
    free_matrix(W, n);

    return py_result;
}


// Map Python method names to our C wrapper functions
static PyMethodDef symnmfMethods[] = {
    {"sym", sym_wrapper, METH_VARARGS, "Calculate the similarity matrix"},
    {"ddg", ddg_wrapper, METH_VARARGS, "Calculate the diagonal degree matrix"},
    {"norm", norm_wrapper, METH_VARARGS, "Calculate the normalized similarity matrix"},
    {"symnmf", symnmf_wrapper, METH_VARARGS, "Perform the SymNMF algorithm"},
    {NULL, NULL, 0, NULL} // Sentinel value marking the end of the array
};

// Define the module
static struct PyModuleDef symnmfmodule = {
    PyModuleDef_HEAD_INIT,
    "symnmf", // Module name in Python (import symnmf)
    "SymNMF C Extension", // Module documentation
    -1,
    symnmfMethods
};

// Initialize the module
PyMODINIT_FUNC PyInit_symnmf(void) {
    return PyModule_Create(&symnmfmodule);
}