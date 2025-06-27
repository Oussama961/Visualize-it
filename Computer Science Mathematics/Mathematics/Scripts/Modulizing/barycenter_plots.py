import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Import the barycenter functions from the core module
from barycenter_core import compute_barycenter, compute_leibniz_function

def configure_seaborn():
    """Configure Seaborn styling for all plots."""
    sns.set_theme(context="notebook", style="whitegrid", palette="deep")
    plt.rcParams["figure.figsize"] = (8, 6)

def plot_barycenter(points, weights, show_leibniz=False, reference_M=None):
    """Plot points, barycenter, and optionally Leibniz vectors."""
    configure_seaborn()
    fig, ax = plt.subplots()
    
    # Plot points
    for i, ((x, y), w) in enumerate(zip(points, weights)):
        ax.scatter(x, y, s=500 * abs(w), label=f'A{i+1} (λ={w})', alpha=0.7)
    
    # Compute barycenter
    G = compute_barycenter(points, weights)
    ax.scatter(*G, s=200, color='black', marker='X', label='Barycenter G')
    
    # Plot Leibniz vectors if requested
    if show_leibniz and reference_M is not None:
        leibniz_vector = compute_leibniz_function(points, weights, reference_M)
        ax.quiver(*reference_M, *leibniz_vector, 
                  angles='xy', scale_units='xy', scale=1,
                  color='red', width=0.005, 
                  label=f'Leibniz Vector at M = {tuple(round(v, 2) for v in leibniz_vector)}')
        
        # Add a line to G for reference
        ax.plot([reference_M[0], G[0]], [reference_M[1], G[1]], 'k--', alpha=0.3)
    
    ax.set_title(f"Barycenter of {len(points)} Points (Total λ = {sum(weights):.1f})")
    ax.legend()
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    plt.grid(True)
    plt.tight_layout()
    
    # Save figure if output directory exists
    output_dir = "../Outputs/Algebra"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/barycenter.png", dpi=300)
    
    return fig 