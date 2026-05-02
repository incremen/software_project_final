"""Python CLI front-end for the symNMF C extension."""
import sys

import numpy as np

import symnmf as symnmf_ext

np.random.seed(1234)

ERR_MSG = "An Error Has Occurred"
VALID_GOALS = ("symnmf", "ddg", "sym", "norm")


def main():
    """Dispatch on the CLI goal and print the resulting matrix."""
    k, goal, file_name = _parse_args()
    data_points, num_points = _read_data(file_name)
    if k >= num_points:
        _die()

    result = _run_goal(goal, data_points, num_points, k)
    for row in result:
        print(",".join(f"{value:.4f}" for value in row))


def _run_goal(goal, data_points, num_points, k):
    """Invoke the appropriate C-extension routine for the given goal."""
    if goal == "sym":
        return symnmf_ext.sym(data_points)
    if goal == "ddg":
        return symnmf_ext.ddg(data_points)
    if goal == "norm":
        return symnmf_ext.norm(data_points)
    # goal == "symnmf"
    norm_sim = symnmf_ext.norm(data_points)
    init_assoc = _initial_assoc(norm_sim, num_points, k)
    return symnmf_ext.symnmf(norm_sim, init_assoc)


def _initial_assoc(norm_sim, num_points, k):
    """Draw H^(0) uniformly from [0, 2*sqrt(mean(W)/k)]."""
    mean_val = np.mean(np.array(norm_sim))
    upper = 2 * np.sqrt(mean_val / k)
    return np.random.uniform(0, upper, size=(num_points, k)).tolist()


def _parse_args():
    """Validate argv and return (k, goal, file_name)."""
    if len(sys.argv) != 4:
        _die()
    try:
        k = int(sys.argv[1])
    except ValueError:
        _die()
    if k <= 1:
        _die()
    goal = sys.argv[2]
    if goal not in VALID_GOALS:
        _die()
    return k, goal, sys.argv[3]


def _read_data(file_name):
    """Parse the input .txt file into (points, row_count)."""
    try:
        with open(file_name, 'r') as fh:
            points = [[float(x) for x in line.strip().split(',')]
                      for line in fh]
        return points, len(points)
    except Exception:
        _die()


def _die():
    """Print the canonical error message and exit non-zero."""
    print(ERR_MSG)
    sys.exit(1)


if __name__ == "__main__":
    main()
