import sys
import numpy as np
import symnmf

np.random.seed(1234)
ERR_MSG = "An Error Has Occurred"


def check_arguments():
    if len(sys.argv) != 4:
        print(ERR_MSG)
        sys.exit(1)
    try:
        k = int(sys.argv[1])
    except ValueError:
        print(ERR_MSG)
        sys.exit(1)
    if k <= 1:
        print(ERR_MSG)
        sys.exit(1)
    goal = sys.argv[2]
    if goal not in ["symnmf", "ddg", "sym", "norm"]:
        print(ERR_MSG)
        sys.exit(1)
    file_name = sys.argv[3]
    return k, goal, file_name

def read_data(file_name):
    try:
        n = 0
        data_points = []
        with open(file_name, 'r') as file:
            for line in file:
                # Strip the newline character, split by comma, and convert each to float
                n += 1
                row = [float(x) for x in line.strip().split(',')]
                data_points.append(row)
        return data_points, n
    except Exception:
        print(ERR_MSG)
        sys.exit(1)


def generate_H(W, n, k):

    W_np = np.array(W)
    
    m = np.mean(W_np)
    
    upper_bound = 2 * np.sqrt(m / k)
    
    H_np = np.random.uniform(0, upper_bound, size=(n, k))
    
    return H_np.tolist()
    

def main():
    k, goal, file_name = check_arguments()
    data_points, n = read_data(file_name)
    if k >= n:
        print(ERR_MSG)
        sys.exit(1)
    result = None
    if goal == "symnmf":
        W = symnmf.norm(data_points)
        H = generate_H(W, n, k)
        result = symnmf.symnmf(W, H)
    elif goal == "ddg":
        result = symnmf.ddg(data_points)
    elif goal == "sym":
        result = symnmf.sym(data_points)
    elif goal == "norm":
        result = symnmf.norm(data_points)
    for row in result:
        print(",".join(f"{v:.4f}" for v in row))

        
if __name__ == "__main__":
    main()