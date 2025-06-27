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