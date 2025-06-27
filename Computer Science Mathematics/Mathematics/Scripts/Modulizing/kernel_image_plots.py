import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ipywidgets import interact

def configure_seaborn():
    sns.set_theme(context="notebook", style="whitegrid", palette="deep")

def plot_kernel_image(f_matrix, title="Kernel and Image of Linear Map"):
    """Visualize Ker(f) and Im(f) for 2D linear map represented by a matrix."""
    configure_seaborn()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    # Domain (E) and Codomain (F) grids
    x = np.linspace(-5, 5, 20)
    X, Y = np.meshgrid(x, x)
    domain = np.vstack([X.ravel(), Y.ravel()]).T
    # Apply linear transformation
    codomain = domain @ f_matrix.T
    # Kernel Visualization (E-space)
    # Solve Ax = 0 (kernel)
    _, _, V = np.linalg.svd(f_matrix)
    kernel_basis = V[-1] if np.linalg.matrix_rank(f_matrix) < 2 else np.zeros(2)
    
    ax1.set_title(f"Ker(f) in E (Dim={1 if kernel_basis.any() else 0})")
    if kernel_basis.any():
        t = np.linspace(-5, 5, 100)
        kernel_line = np.outer(t, kernel_basis)
        ax1.plot(kernel_line[:,0], kernel_line[:,1], 'r--', label="Ker(f)")
    ax1.scatter(domain[:,0], domain[:,1], alpha=0.1, color='blue')
    ax1.legend()
    # Image Visualization (F-space)
    ax2.set_title(f"Im(f) in F (Dim={np.linalg.matrix_rank(f_matrix)})")
    ax2.scatter(codomain[:,0], codomain[:,1], alpha=0.1, color='green')    
    # Span of Im(f)
    im_basis = f_matrix[:,0], f_matrix[:,1]
    t = np.linspace(-5, 5, 100)
    im_line = t[:,None] @ np.array(im_basis).T
    ax2.plot(im_line[:,0], im_line[:,1], 'm--', label="Im(f)")
    ax2.legend()
    plt.suptitle(title)
    plt.show()

# Interactive widget for matrix entries
@interact(a=(-2.0, 2.0), b=(-2.0, 2.0), c=(-2.0, 2.0), d=(-2.0, 2.0))
def update_kernel_image(a=1, b=0, c=0, d=1):
    plot_kernel_image(np.array([[a, b], [c, d]]))


def plot_injectivity_test(f_matrix):
    """Check if Ker(f) = {0} (injective)."""
    configure_seaborn()
    fig, ax = plt.subplots(figsize=(6, 6))
    # Generate non-zero vectors
    angles = np.linspace(0, 2*np.pi, 20)
    vectors = np.array([np.cos(angles), np.sin(angles)]).T
    # Apply transformation
    transformed = vectors @ f_matrix.T
    # Plot pre- and post-transformation
    ax.quiver(0, 0, vectors[:,0], vectors[:,1], color='blue', alpha=0.5, label="Original Vectors")
    ax.quiver(0, 0, transformed[:,0], transformed[:,1], color='red', alpha=0.5, label="Transformed Vectors")
    # Check injectivity
    kernel_dim = np.linalg.matrix_rank(f_matrix)
    is_injective = (kernel_dim == 0) if f_matrix.shape[0] == f_matrix.shape[1] else False
    ax.set_title(f"Injective: {is_injective} (Ker(f) Dim={kernel_dim})")
    ax.legend()
    plt.show()


def plot_subspace_image(f_matrix, subspace_angle):
    """Visualize f(A) for a subspace A of E."""
    configure_seaborn()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    # Original subspace A in E
    t = np.linspace(-5, 5, 100)
    A = np.outer(t, [np.cos(subspace_angle), np.sin(subspace_angle)])
    ax1.plot(A[:,0], A[:,1], 'b--', label="Subspace A ⊂ E")
    # Image f(A) in F
    f_A = A @ f_matrix.T
    ax2.plot(f_A[:,0], f_A[:,1], 'r--', label="f(A) ⊂ F")
    ax1.set_title("Original Subspace (E-space)")
    ax2.set_title("Image Subspace (F-space)")
    ax1.legend()
    ax2.legend()
    plt.show()

def plot_preimage(f_matrix, B_angle):
    """Visualize f⁻¹(B) for a subspace B of F."""
    configure_seaborn()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    # Subspace B in F
    t = np.linspace(-5, 5, 100)
    B = np.outer(t, [np.cos(B_angle), np.sin(B_angle)])
    ax2.plot(B[:,0], B[:,1], 'g--', label="Subspace B ⊂ F")
    # Preimage f⁻¹(B) in E (approximated)
    try:
        f_inv = np.linalg.inv(f_matrix)
        preimage = B @ f_inv.T
        ax1.plot(preimage[:,0], preimage[:,1], 'm--', label="f⁻¹(B) ⊂ E")
    except np.linalg.LinAlgError:
        print("Non-invertible map! Preimage may not be a subspace.")
    ax1.set_title("Preimage Subspace (E-space)")
    ax2.set_title("Original Subspace (F-space)")
    plt.show()