"""K-means clustering (HW1 implementation, kept compatible with analysis.py)."""
import math
import sys

K_ERROR_MSG = "Incorrect number of clusters!"
ITER_ERROR_MSG = "Incorrect maximum iteration!"
GENERAL_ERROR_MSG = "An Error Has Occurred"
EPSILON = 0.001


def main():
    """CLI: read vectors from stdin, print the final centroids."""
    is_valid, k, max_iter, vectors = _parse_cli_args()
    if not is_valid:
        return
    centroids = kmeans(vectors, k, max_iter)
    for centroid in centroids:
        print(",".join(f"{coord:.4f}" for coord in centroid))


def kmeans(vectors, k, max_iter):
    """Classic Lloyd's algorithm; returns the converged list of centroids."""
    centroids = _initialize_centroids(vectors, k)
    for _ in range(max_iter):
        clusters = _assign_clusters(vectors, centroids)
        updated = _update_centroids(clusters, vectors)
        if _has_converged(centroids, updated):
            return updated
        centroids = updated
    return centroids


def euclidean_distance(left, right):
    """L2 distance between two equally sized iterables of floats."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def _initialize_centroids(vectors, k):
    """Seed centroids with the first k vectors, as specified in HW1."""
    return [tuple(v) for v in vectors[:k]]


def _assign_clusters(vectors, centroids):
    """Group each vector with its nearest centroid."""
    clusters = [[] for _ in centroids]
    for vector in vectors:
        distances = [euclidean_distance(vector, c) for c in centroids]
        clusters[distances.index(min(distances))].append(vector)
    return clusters


def _update_centroids(clusters, vectors):
    """Mean of each cluster; empty clusters fall back to vectors[0]."""
    updated = []
    for cluster in clusters:
        if not cluster:
            updated.append(vectors[0])
            continue
        size = len(cluster)
        dim = len(cluster[0])
        updated.append(tuple(sum(p[axis] for p in cluster) / size
                             for axis in range(dim)))
    return updated


def _has_converged(previous, current):
    """True once every centroid has moved less than EPSILON."""
    return all(euclidean_distance(a, b) < EPSILON
               for a, b in zip(previous, current))


def _parse_k(argv, num_vectors):
    """Return (ok, k) after validating the CLI k argument."""
    if len(argv) < 2:
        print(K_ERROR_MSG)
        return False, None
    try:
        k = int(argv[1])
    except ValueError:
        print(K_ERROR_MSG)
        return False, None
    if not (1 < k < num_vectors):
        print(K_ERROR_MSG)
        return False, None
    return True, k


def _parse_iter(argv):
    """Return (ok, max_iter) after validating the optional iter argument."""
    if len(argv) == 2:
        return True, 400
    try:
        max_iter = int(argv[2])
    except ValueError:
        print(ITER_ERROR_MSG)
        return False, None
    if not (1 < max_iter < 800):
        print(ITER_ERROR_MSG)
        return False, None
    return True, max_iter


def _read_vectors_from_stdin():
    """Parse comma-separated floats from stdin into tuples."""
    return [tuple(float(coord) for coord in line.split(','))
            for line in sys.stdin.read().strip().split('\n') if line.strip()]


def _parse_cli_args():
    """Combined input parsing: argv + stdin vectors."""
    if len(sys.argv) > 3:
        print(GENERAL_ERROR_MSG)
        return False, None, None, None
    vectors = _read_vectors_from_stdin()
    ok_k, k = _parse_k(sys.argv, len(vectors))
    ok_iter, max_iter = _parse_iter(sys.argv)
    if not (ok_k and ok_iter):
        return False, None, None, None
    return True, k, max_iter, vectors


if __name__ == "__main__":
    main()
