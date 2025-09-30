import torch
import numpy as np
import os

device = (
    torch.device("mps") if torch.backends.mps.is_available() 
    else torch.device("cuda") if torch.cuda.is_available() 
    else torch.device("cpu")
)

print("The current device is", device)

def generate_data(n=2000, K=3, seed=123, save=False):
    """
    Generate 2D Gaussian mixture data and save as a single CSV (optional).
    
    Parameters:
    - n: total number of samples
    - K: number of Gaussian components
    - seed: random seed for reproducibility
    
    Saves:
    - data.csv : shape (n, 3) where columns are x1, x2, label
    """
    torch.manual_seed(seed)
    
    means = (torch.rand(K, 2, device=device) * 10 - 5)  # random in [-5, 5]
    covs = torch.stack([torch.eye(2, device=device) for _ in range(K)])
    weights = torch.ones(K, device=device) / K

    counts = torch.multinomial(weights, n, replacement=True)
    counts = torch.bincount(counts, minlength=K)
    
    X_list = []
    y_list = []
    
    for k in range(K):
        mean = means[k]
        cov = covs[k]
        L = torch.linalg.cholesky(cov)
        z = torch.randn(counts[k], 2, device=device)
        X_k = z @ L.T + mean
        X_list.append(X_k)
        y_list.append(torch.full((counts[k],), k, device=device, dtype=torch.long))

    X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list, dim=0)
    
    if not save:
        return X, y

    # Combine X and y into a single array (last column is label)
    data = torch.cat([X, y.unsqueeze(1)], dim=1).cpu().numpy()

    # Make sure save directory exists
    save_dir="data/simulated"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save to CSV with header
    np.savetxt(os.path.join(save_dir, "data.csv"), data, delimiter=",", header="x1,x2,label", comments="", fmt=["%.5f","%.5f","%d"])
    
    print(f"Generated {n} samples with {K} classes saved to {save_dir}/data.csv")

    return X, y

# Example usage:
generate_data(n=5000, K=4, save=True)