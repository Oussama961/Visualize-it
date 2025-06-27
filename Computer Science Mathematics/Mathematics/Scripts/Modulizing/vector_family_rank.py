import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ipywidgets import interact

def configure_seaborn():
    sns.set_theme(context="notebook", style="whitegrid", palette="deep")

def plot_vector_family_rank(vectors):
    """Visualize the subspace spanned by vectors and compute its rank."""
    configure_seaborn()
    fig, ax = plt.subplots()
    
    # Plot vectors
    colors = sns.color_palette("husl", len(vectors))
    for i, (x, y) in enumerate(vectors):
        ax.quiver(0, 0, x, y, angles='xy', scale_units='xy', scale=1,
                  color=colors[i], width=0.02, label=f'v{i+1} = ({x}, {y})')
    
    # Compute rank (dimension of the spanned subspace)
    matrix = np.array(vectors).T
    rank = np.linalg.matrix_rank(matrix)
    
    # Highlight spanned subspace
    if rank == 1:
        # Line spanned by the first vector
        t = np.linspace(-5, 5, 100)
        line = t[:, None] * vectors[0]
        ax.plot(line[:,0], line[:,1], 'r--', alpha=0.5, label="Span (Rank=1)")
    elif rank == 2:
        # Fill the plane (for 2D)
        ax.set_facecolor(sns.color_palette("pastel")[0])
        ax.set_title("Span = ℝ² (Rank=2)")
    
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.legend()
    plt.title(f"Rank of Vector Family = {rank}")
    plt.show()

# Interactive widget for 2 vectors
@interact(v1_x=(-2.0, 2.0), v1_y=(-2.0, 2.0), 
          v2_x=(-2.0, 2.0), v2_y=(-2.0, 2.0))
def update_vector_family(v1_x=1, v1_y=2, v2_x=3, v2_y=1):
    plot_vector_family_rank([(v1_x, v1_y), (v2_x, v2_y)])


# linear_map_rank _____
def plot_linear_map_rank(matrix):
    """Visualize rank and kernel of a linear map (2x2 matrix)."""
    configure_seaborn()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    # Kernel Visualization (E-space)
    # Solve Ax = 0
    # where A is the matrix and x is the vector in E (ℝ²)
    # The kernel is the null space of the matrix
    _, _, V = np.linalg.svd(matrix)
    kernel_basis = V[-1] if np.linalg.matrix_rank(matrix) < 2 else np.zeros(2)
    rank = np.linalg.matrix_rank(matrix)
    
    ax1.set_title(f"Kernel (Dim={2 - rank})")
    if np.any(kernel_basis):
        t = np.linspace(-5, 5, 100)
        kernel_line = t[:, None] * kernel_basis
        ax1.plot(kernel_line[:,0], kernel_line[:,1], 'r--', label="Ker(u)")
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(-5, 5)
    ax1.legend()
    ax2.set_title(f"Image (Rank={rank})")
    # Plot column vectors (image basis)
    ax2.quiver(0, 0, matrix[0,0], matrix[1,0], color='blue', scale=10, label="u(e₁)")
    ax2.quiver(0, 0, matrix[0,1], matrix[1,1], color='green', scale=10, label="u(e₂)")
    ax2.set_xlim(-5, 5)
    ax2.set_ylim(-5, 5)
    ax2.legend()    
    plt.suptitle(f"Matrix: {matrix}\nRank = {rank}, Kernel Dim = {2 - rank}")
    plt.show()

# rank_nullity

def plot_rank_nullity(matrix):
    """Visualize rank-nullity theorem: dim(E) = rank(u) + dim(ker(u))."""
    configure_seaborn()
    rank = np.linalg.matrix_rank(matrix)
    kernel_dim = 2 - rank  # For 2x2 matrices
    
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.text(0.5, 0.8, f"dim(E) = {2}", ha='center', fontsize=12)
    ax.text(0.5, 0.6, f"rank(u) = {rank}", ha='center', fontsize=12)
    ax.text(0.5, 0.4, f"dim(ker(u)) = {kernel_dim}", ha='center', fontsize=12)
    ax.text(0.5, 0.2, f"2 = {rank} + {kernel_dim}", ha='center', fontsize=14, color='red')
    plt.title("Rank-Nullity Theorem: dim(E) = rank(u) + dim(ker(u))")
    plt.show()

def plot_bijective_equivalence(matrix):
    """Show equivalence of injectivity/surjectivity when dim(E) = dim(F)."""
    configure_seaborn()
    rank = np.linalg.matrix_rank(matrix)
    is_injective = (rank == 2)
    is_surjective = (rank == 2)
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.text(0.5, 0.8, f"dim(E) = dim(F) = 2", ha='center', fontsize=12)
    ax.text(0.5, 0.6, f"Injective: {is_injective}", ha='center', fontsize=12)
    ax.text(0.5, 0.4, f"Surjective: {is_surjective}", ha='center', fontsize=12)
    ax.text(0.5, 0.2, f"Bijective: {is_injective and is_surjective}", 
            ha='center', fontsize=14, color='blue')
    plt.title("Corollary: Injective ⇔ Surjective ⇔ Bijective")
    plt.show()

