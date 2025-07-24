import torch

def generate_batched_permutations(n_perm: int, n: int, device: torch.device = "cpu") -> torch.Tensor:
    """
    Generate a batch of random permutations.

    Args:
        n_perm (int): Number of permutations to generate (batch size).
        n (int): Size of each permutation.
        device (torch.device or str): Device for the resulting tensor.

    Returns:
        Tensor of shape (n_perm, n), where each row is a random permutation of [0, ..., n-1].
    """
    base = torch.arange(n, device=device)
    perms = torch.stack([base[torch.randperm(n)] for _ in range(n_perm)])
    return perms


def batched_flat_permutation(perm_batch: torch.Tensor) -> torch.Tensor:
    """
    Given a batch of permutations of shape (n_perms, n), compute
    the 'flat-permutation' index array of shape (n_perms, N), where
    N = n*(n-1)//2.

    - perm_batch[p] is a permutation of [0..n-1].
    - The returned array, say 'order', satisfies:
         flat_reordered = flat_original[ order[p] ]
      which is equivalent to:
         1) reshaping flat_original back to (n x n) (upper tri only)
         2) permuting rows/columns of that n x n with perm_batch[p]
         3) flattening the upper triangle again.
    
    This version avoids an O(N log N) sort, instead building the
    new->old mapping in O(N) time for each permutation.
    """

    # Shape info
    n_perms, n = perm_batch.shape
    device = perm_batch.device
    # Number of strictly upper-triangle entries
    N = n * (n - 1) // 2

    # 1) Build inverse permutation for each p in perm_batch
    #    invPerm[p, perm_batch[p, k]] = k
    invPerm = torch.empty_like(perm_batch)
    rows = torch.arange(n_perms, device=device).unsqueeze(1).expand(-1, n)  # (n_perms, n)
    cols = perm_batch
    vals = torch.arange(n, device=device).unsqueeze(0).expand(n_perms, -1)  # (n_perms, n)
    invPerm[rows, cols] = vals

    # 2) Precompute all (i, j) for upper triangle (i < j)
    i, j = torch.triu_indices(n, n, offset=1, device=device)

    # 3) For each permutation p, map (i, j) -> (new_i, new_j)
    new_i = invPerm[:, i]  # (n_perms, N)
    new_j = invPerm[:, j]  # (n_perms, N)

    # 4) Ensure new_i < new_j
    lower = torch.min(new_i, new_j)
    upper = torch.max(new_i, new_j)

    # 5) "old->new" positions: new_indices[p, k] = new position of old index k
    tmp = (n - lower) * (n - lower - 1) // 2
    new_indices = (N - tmp) + (upper - lower - 1)  # shape (n_perms, N)

    # 6) Build "new->old" order in O(N) time (no sort):
    #    order[p, new_pos] = old_pos
    order = torch.empty_like(new_indices)
    old_positions = torch.arange(N, device=device).unsqueeze(0).expand(n_perms, -1)  # (n_perms, N)
    row_ids = torch.arange(n_perms, device=device).unsqueeze(1).expand(-1, N)        # (n_perms, N)

    order[row_ids, new_indices] = old_positions  # scatter: new->old

    return order



def flat_permutation(perm: torch.Tensor) -> torch.Tensor:
    """
    Given a single permutation of shape (n,), compute the 'flat-permutation'
    index array of shape (n*(n-1)//2,), such that:

        flat_reordered = flat_original[ flat_permutation_no_sort(perm) ]

    is equivalent to:
        matrix -> permuted_matrix = matrix[perm][:, perm]
        flat_reordered = upper_triangle(permuted_matrix)

    The result maps new indices (after permutation) back to the original order,
    constructed without sorting, in O(N) time.
    """
    n = perm.size(0)
    device = perm.device
    N = n * (n - 1) // 2

    # Inverse permutation
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(n, device=device)

    # Get upper-triangular indices
    i, j = torch.triu_indices(n, n, offset=1, device=device)

    # Map (i, j) -> (inv_perm[i], inv_perm[j])
    new_i = inv_perm[i]
    new_j = inv_perm[j]

    lower = torch.minimum(new_i, new_j)
    upper = torch.maximum(new_i, new_j)

    tmp = (n - lower) * (n - lower - 1) // 2
    new_indices = (N - tmp) + (upper - lower - 1)

    # Build new->old map in O(N)
    order = torch.empty(N, dtype=torch.long, device=device)
    order[new_indices] = torch.arange(N, device=device)

    return order
