import numpy as np

def compute_barycenter(points, weights):
    """
    Compute barycenter of weighted points.
    - points: List of tuples [(x1, y1), (x2, y2), ...]
    - weights: List of weights [λ1, λ2, ...]
    Returns barycenter coordinates (Gx, Gy).
    """
    total_weight = sum(weights)
    if np.isclose(total_weight, 0):
        raise ValueError("Total weight cannot be zero!")
    
    weighted_sum = np.sum([w * np.array(p) for w, p in zip(weights, points)], axis=0)
    return tuple(weighted_sum / total_weight)

def compute_leibniz_function(points, weights, M):
    """
    Compute Leibniz vector function ƒ(M) = Σλ_i·MA_i.
    - M: Reference point (tuple)
    Returns vector sum.
    """
    return np.sum([w * (np.array(p) - np.array(M)) for w, p in zip(weights, points)], axis=0) 