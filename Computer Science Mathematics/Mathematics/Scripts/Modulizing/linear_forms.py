def plot_linear_form(phi, title="Linear Form φ: E → K"):
    """Visualize a linear functional as contour lines."""
    configure_seaborn()
    fig, ax = plt.subplots()
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = phi(np.stack([X.ravel(), Y.ravel()]).T.reshape(X.shape)
    cs = ax.contour(X, Y, Z, levels=10, cmap='viridis')
    ax.clabel(cs, inline=True)
    ax.set_title(title)
    plt.show()

def phi_example(v):
    return 2*v[0] + 3*v[1]

plot_linear_form(phi_example)