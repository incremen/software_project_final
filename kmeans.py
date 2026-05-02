import sys
import math


K_ERROR_MSG = "Incorrect number of clusters!"
ITER_ERROR_MSG = "Incorrect maximum iteration!"
GENERAL_ERROR_MSG = "An Error Has Occurred"
EPSILON = 0.001


def get_K(argv, N):
    """
    Returns (is_valid, K)
    """
    if len(argv) < 2:
        print(K_ERROR_MSG)
        return False, None
    K_string = argv[1]
    try:
        K = int(K_string)
    except ValueError:
        print(K_ERROR_MSG)
        return False, None

    if not (1 < K < N):
        print(K_ERROR_MSG)
        return False, None
    
    return True, K


def get_iter(argv):
    """
    Returns (is_valid, iter)
    """
    if len(argv) == 2:
        return True, 400  # default value
    
    iter_string = argv[2]
    try:
        iter = int(iter_string)
    except ValueError:
        print(ITER_ERROR_MSG)
        return False, None
    
    if not (1 < iter < 800):
        print(ITER_ERROR_MSG)
        return False, None
    
    return True, iter



def read_vectors():
    """
    Reads data from stdin and returns a list of tuples.
    Each line is parsed as comma-separated floats.
    """
    lines = sys.stdin.read().strip().split('\n')
    data = []
    for line in lines:
        if line.strip():
            coords = line.split(',')
            point = tuple(float(coord) for coord in coords)
            data.append(point)
    return data


def get_input():
    """
    Returns a tuple: (is_valid, K, iterations, data)
    Also reads data from stdin.
    """
    argc = len(sys.argv)
    argv = sys.argv
    
    if argc > 3:
        print(GENERAL_ERROR_MSG)
        return False, None, None, None

    vectors = read_vectors()
    N = len(vectors)

    is_K_valid, K = get_K(argv, N)
    is_iter_valid, iter_val = get_iter(argv)

    if not (is_K_valid and is_iter_valid):
        return False, None, None, None
    

    return True, K, iter_val, vectors


def euclidean_distance(p, q):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p, q)))


def initialize_centroids(vectors, K):
    return [tuple(v) for v in vectors[:K]]


def assign_clusters(vectors, centroids):
    clusters = [[] for _ in centroids]
    for v in vectors:
        distance_from_clusters = [euclidean_distance(v, c) for c in centroids]
        closest_cluster_idx = distance_from_clusters.index(min(distance_from_clusters))
        clusters[closest_cluster_idx].append(v)
    return clusters


def update_centroids(clusters, prev_centroids, vectors):
    new_centroids = []
    for i, cluster in enumerate(clusters):

        if not cluster: # empty cluster
            new_centroids.append(vectors[0]) # reassign to first vector
        else:
            n = len(cluster)
            vectors_dim = len(cluster[0])
            centroid = tuple(sum(p[j] for p in cluster) / n for j in range(vectors_dim))
            new_centroids.append(centroid)
    return new_centroids


def centroids_have_converged(prev, new):
    return all(euclidean_distance(a, b) < EPSILON for a, b in zip(prev, new))


def kmeans(vectors, K, max_iter):
    centroids = initialize_centroids(vectors, K)
    for _ in range(max_iter):
        clusters = assign_clusters(vectors, centroids)
        new_centroids = update_centroids(clusters, centroids, vectors)

        if centroids_have_converged(centroids, new_centroids):
            centroids = new_centroids
            break
        
        centroids = new_centroids
    return centroids


def main():
    # print("starting")
    is_input_valid, K, iterations, vectors = get_input()
    if not is_input_valid:
        return
    # print(f"{is_input_valid=}, {K=}, {iterations=}, {len(vectors)=}")

    centroids = kmeans(vectors, K, iterations)
    for c in centroids:
        print(",".join(f"{coord:.4f}" for coord in c))

if __name__ == "__main__":
    main()



    
    
