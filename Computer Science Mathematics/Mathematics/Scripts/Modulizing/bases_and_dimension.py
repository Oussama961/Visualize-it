from mpl_toolkits.mplot3d import Axes3D

def plot_basis_3d(v1, v2, v3):
    """Show 3D basis vectors and their combinations."""
    configure_seaborn()
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot vectors
    ax.quiver(0, 0, 0, v1[0], v1[1], v1[2], color='blue', label=f'v1 = {v1}')
    ax.quiver(0, 0, 0, v2[0], v2[1], v2[2], color='green', label=f'v2 = {v2}')
    ax.quiver(0, 0, 0, v3[0], v3[1], v3[2], color='red', label=f'v3 = {v3}')
    
    # Check if basis (determinant ≠ 0)
    matrix = np.array([v1, v2, v3])
    is_basis = not np.isclose(np.linalg.det(matrix), 0)
    ax.set_title(f"Basis of ℝ³: {is_basis}")
    
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    ax.legend()
    plt.show()

@interact(
    v1_x=(-2.0, 2.0), v1_y=(-2.0, 2.0), v1_z=(-2.0, 2.0),
    v2_x=(-2.0, 2.0), v2_y=(-2.0, 2.0), v2_z=(-2.0, 2.0),
    v3_x=(-2.0, 2.0), v3_y=(-2.0, 2.0), v3_z=(-2.0, 2.0)
)
def update_basis(
    v1_x=1, v1_y=0, v1_z=0,
    v2_x=0, v2_y=1, v2_z=0,
    v3_x=0, v3_y=0, v3_z=1
):
    plot_basis_3d((v1_x, v1_y, v1_z), (v2_x, v2_y, v2_z), (v3_x, v3_y, v3_z))