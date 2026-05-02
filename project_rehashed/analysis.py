"""Compare symNMF clustering to k-means via silhouette score."""
import sys

import numpy as np
from sklearn.metrics import silhouette_score

import kmeans
import symnmf as symnmf_ext

np.random.seed(1234)

ERR_MSG = "An Error Has Occurred"


def main():
    """CLI: print silhouette scores for both algorithms on the given file."""
    try:
        k, file_name = _parse_args()
        data_np = np.loadtxt(file_name, delimiter=',')
        data_list = data_np.tolist()

        nmf_labels = _symnmf_labels(data_list, len(data_list), k)
        kmeans_labels = _kmeans_labels(data_np, k)

        print(f"nmf: {silhouette_score(data_np, nmf_labels):.4f}")
        print(f"kmeans: {silhouette_score(data_np, kmeans_labels):.4f}")
    except Exception:
        print(ERR_MSG)
        sys.exit(1)


def _symnmf_labels(data_list, num_points, k):
    """Run symNMF and pick each row's argmax as its cluster assignment."""
    norm_sim = symnmf_ext.norm(data_list)
    init_assoc = _initial_assoc(norm_sim, num_points, k)
    final_assoc = symnmf_ext.symnmf(norm_sim, init_assoc)
    return np.argmax(np.array(final_assoc), axis=1)


def _kmeans_labels(data_np, k):
    """Run k-means (HW2 convergence settings) and return per-point labels."""
    kmeans.EPSILON = 1e-4
    vectors = [tuple(row) for row in data_np]
    centroids = kmeans.kmeans(vectors, k, 300)
    return [min(range(len(centroids)),
                key=lambda idx: kmeans.euclidean_distance(v, centroids[idx]))
            for v in vectors]


def _initial_assoc(norm_sim, num_points, k):
    """H^(0) sampled uniformly from [0, 2*sqrt(mean(W)/k)] (mirrors symnmf.py)."""
    mean_val = np.mean(np.array(norm_sim))
    upper = 2 * np.sqrt(mean_val / k)
    return np.random.uniform(0, upper, size=(num_points, k)).tolist()


def _parse_args():
    """Validate argv and return (k, file_name)."""
    if len(sys.argv) != 3:
        raise ValueError("usage: analysis.py <k> <file>")
    return int(sys.argv[1]), sys.argv[2]


if __name__ == "__main__":
    main()
