import sys
import numpy as np
from sklearn.metrics import silhouette_score

# Safely import your HW1 logic
import kmeans 
# Import the compiled C-extension (not your symnmf.py script)
import symnmf

# Required by assignment instructions
np.random.seed(1234)

def init_H(W_list, n, k):
    """Replicated from your symnmf.py to avoid import collision"""
    W_np = np.array(W_list)
    m = np.mean(W_np)
    upper_bound = 2 * np.sqrt(m / k)
    H_np = np.random.uniform(0, upper_bound, size=(n, k))
    return H_np.tolist()

def main():
    print("Running Analysis...")
    if len(sys.argv) != 3:
        print("An Error Has Occurred")
        sys.exit(1)
        
    try:
        k = int(sys.argv[1])
        file_name = sys.argv[2]
        
        # 1. Read the data
        data_np = np.loadtxt(file_name, delimiter=',')
        data_list = data_np.tolist()
        
        # Convert to a list of tuples exactly as your kmeans.py expects
        vectors = [tuple(row) for row in data_np]
        n = len(vectors)
        
        # ---------------------------------------------------------
        # 2. Run K-Means (Using your kmeans.py!)
        # ---------------------------------------------------------
        # Override the hardcoded EPSILON in kmeans.py to match HW2 specs
        kmeans.EPSILON = 1e-4 
        
        # Call your algorithm (max_iter = 300 per HW2 specs)
        centroids = kmeans.kmeans(vectors, k, 300)
        
        # silhouette_score needs a list of labels (cluster indices) for each point.
        # Your kmeans returns centroids, so we quickly map each point to its closest centroid.
        kmeans_labels = []
        for v in vectors:
            distances = [kmeans.euclidean_distance(v, c) for c in centroids]
            kmeans_labels.append(distances.index(min(distances)))

        # ---------------------------------------------------------
        # 3. Run SymNMF
        # ---------------------------------------------------------
        W = symnmf.norm(data_list)
        H = init_H(W, n, k)
        
        # Calling your C-extension exactly as you did in symnmf.py
        final_H = symnmf.symnmf(W, H)
        
        # Derive hard clustering (index of max value in each row)
        nmf_labels = np.argmax(np.array(final_H), axis=1)

        # ---------------------------------------------------------
        # 4. Calculate and Print Scores
        # ---------------------------------------------------------
        nmf_score = silhouette_score(data_np, nmf_labels)
        kmeans_score = silhouette_score(data_np, kmeans_labels)
        
        print(f"nmf: {nmf_score:.4f}")
        print(f"kmeans: {kmeans_score:.4f}")

    except Exception:
        print("An Error Has Occurred")
        sys.exit(1)

if __name__ == "__main__":
    main()