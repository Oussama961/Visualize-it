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