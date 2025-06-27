def plot_automorphism(f, inv_f, title="Automorphism (Bijective Endomorphism)"):
    """Visualize an automorphism and its inverse."""
    configure_seaborn()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    x = np.linspace(-5, 5, 5)
    X, Y = np.meshgrid(x, x)
    grid = np.stack([X.ravel(), Y.ravel()]).T
    ax[0].scatter(grid[:,0], grid[:,1], color='blue', alpha=0.5, label="Original Grid")    
    transformed = np.array([f(v) for v in grid])
    ax[0].scatter(transformed[:,0], transformed[:,1], color='red', alpha=0.5, label="Transformed Grid")
    ax[0].set_title("f: E → E (Endomorphism)")
    ax[0].legend()
    inverted = np.array([inv_f(v) for v in transformed])
    ax[1].scatter(inverted[:,0], inverted[:,1], color='green', alpha=0.5, label="Inverse Transformed Grid")
    ax[1].set_title("f⁻¹: E → E (Inverse)")
    ax[1].legend()
    plt.suptitle(title)
    plt.show()

# Example: Rotation (bijective endomorphism)
theta = np.pi/4
def rotation_f(x):
    return np.dot([[np.cos(theta), -np.sin(theta)], 
                  [np.sin(theta), np.cos(theta)]], x)

def rotation_inv(x):
    return np.dot([[np.cos(theta), np.sin(theta)], 
                  [-np.sin(theta), np.cos(theta)]], x)

plot_automorphism(rotation_f, rotation_inv)