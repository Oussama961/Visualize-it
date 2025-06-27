import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ipywidgets import interact

def configure_seaborn():
    sns.set_theme(context="notebook", style="whitegrid", palette="deep")

def plot_linear_map_from_basis(e1_image, e2_image):
    """Visualize linear map defined by images of standard basis vectors."""
    configure_seaborn()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    # Original basis in E (ℝ²)
    ax1.quiver(0, 0, 1, 0, color='blue', scale=10, label='e₁')
    ax1.quiver(0, 0, 0, 1, color='green', scale=10, label='e₂')
    ax1.set_title("Original Basis (E)")
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.legend()
    # Transformed basis in F (ℝ²)
    ax2.quiver(0, 0, *e1_image, color='blue', scale=10, label='u(e₁)')
    ax2.quiver(0, 0, *e2_image, color='green', scale=10, label='u(e₂)')
    ax2.set_title("Image Basis (F)")
    ax2.set_xlim(-5, 5)
    ax2.set_ylim(-5, 5)
    ax2.legend()
    # Matrix representation and properties
    matrix = np.column_stack([e1_image, e2_image])
    determinant = np.linalg.det(matrix)
    rank = np.linalg.matrix_rank(matrix)
    is_injective = (determinant != 0) if matrix.shape[0] == matrix.shape[1] else (rank == len(e1_image))
    is_surjective = (rank == matrix.shape[0])
    plt.suptitle(
        f"Matrix: {matrix}\n"
        f"Det: {determinant:.2f}, Rank: {rank}\n"
        f"Injective: {is_injective}, Surjective: {is_surjective}"
    )
    plt.show()
# Interactive widget for adjusting basis images
@interact(e1_x=(-2.0, 2.0), e1_y=(-2.0, 2.0), e2_x=(-2.0, 2.0), e2_y=(-2.0, 2.0))
def update_map(e1_x=1, e1_y=0, e2_x=0, e2_y=1):
    plot_linear_map_from_basis((e1_x, e1_y), (e2_x, e2_y))


# matrix_visualization _____
def plot_transformation_grid(e1_image, e2_image):
    """Show grid transformation by linear map defined on basis."""
    configure_seaborn()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    # Original grid in E
    x = np.linspace(-5, 5, 10)
    X, Y = np.meshgrid(x, x)
    grid = np.stack([X.ravel(), Y.ravel()]).T
    ax1.scatter(grid[:,0], grid[:,1], alpha=0.3, color='blue')
    ax1.set_title("Original Grid (E)")
    # Transformed grid in F
    matrix = np.column_stack([e1_image, e2_image])
    transformed = grid @ matrix.T
    ax2.scatter(transformed[:,0], transformed[:,1], alpha=0.3, color='red')
    ax2.set_title("Transformed Grid (F)")
    plt.suptitle(f"Linear Map Matrix:\n{matrix}")
    plt.show()

def plot_injective_surjective_test(e1_image, e2_image):
    """Test if u is injective/surjective using basis images."""
    configure_seaborn()
    matrix = np.column_stack([e1_image, e2_image])
    
    # Injectivity: Check if basis images are linearly independent
    is_injective = (np.linalg.matrix_rank(matrix) == 2)
    
    # Surjectivity: Check if basis images span F (ℝ²)
    is_surjective = (np.linalg.matrix_rank(matrix) == 2)
    
    # Visualize linear independence and spanning
    fig, ax = plt.subplots()
    ax.quiver(0, 0, *e1_image, color='blue', scale=10, label='u(e₁)')
    ax.quiver(0, 0, *e2_image, color='green', scale=10, label='u(e₂)')
    ax.set_title(
        f"u is {'bijective' if is_injective and is_surjective else 'injective' if is_injective else 'surjective' if is_surjective else 'neither'}\n"
        f"Independent: {is_injective}, Spanning: {is_surjective}"
    )
    ax.legend()
    plt.show()